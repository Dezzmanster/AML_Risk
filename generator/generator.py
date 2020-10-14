import pandas as pd
import networkx as nx
import gc
import os
import warnings
import featuretools as ft
from utils import timeit, check_path, check_dir, check_csv, \
    check_csv_files, check_col_in_df, save_dataframe_to_csv, date_parser
import logging.config
import logging
gc.enable()
warnings.filterwarnings('ignore')
logging.config.fileConfig(fname='logger.ini', defaults={'logfilename': 'logfile.log'})


class FeatureGenerator(object):

    @timeit
    def __init__(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError('"parameters" must be a dict type')

        if 'path_data' not in parameters.keys():
            raise KeyError('"path_data" is not in "parameters", "path_data" is a necessary parameter')

        if not check_path(parameters['path_data']) or not check_dir(parameters['path_data']):
            raise ValueError(f"'{parameters['path_data']}' does not exists or is not folder")

        if not check_csv(parameters['path_data']):
            raise ValueError(f"'{parameters['path_data']}' must contain only csv files")

        all_file_founded, list_not_founded_files = check_csv_files(parameters['path_data'], parameters['tables'])
        if not all_file_founded:
            raise ValueError(f"{list_not_founded_files} is not founded in {parameters['path_data']}")

        self.path = parameters['path_data']
        self.tables = parameters['tables']
        self.main_table = parameters['main_table']
        self.depth = parameters['depth']
        self.relations = parameters['relations']
        self.sep = parameters['sep']
        self.n_jobs = min(os.cpu_count(), parameters['n_jobs'])
        self.chunk_size = parameters['chunk_size']
        self.max_features = parameters['max_features']
        self.target = parameters['target']
        self.paths_files = [os.path.join(self.path, table) for table, _ in self.tables.items()]
        self.drop_contains = [col for _, col in self.tables.items() if col] + [self.target + ')']
        self.dict_dataframes = dict()
        self.entities = None
        self.agg_primitives = parameters['agg_primitives']
        self.trans_primitives = parameters['trans_primitives']
        self.feature_matrix = None
        self.feature_names = None
        self.output_file_name = parameters['output_file_name']
        self.time_indecies = parameters['time_indecies']
        self.time_variables = parameters['time_variables']  # TODO заменить на обработку из файла Ани

        logging.info(f"Object {self} is created")

    def __repr__(self):
        return f"Generator(data='{self.path}', output_file='{self.output_file_name}', object_id={id(self)})"

    @timeit
    def create_dataframes(self):
        """
        Make DataFrame objects from csv files
        """
        for (file, col), path in zip(self.tables.items(), self.paths_files):
            try:
                parse_dates = self.time_variables[file]  # TODO поменять код на просмотре столбцов из таблицы типов от Ани, тип данных должен быть list
            except KeyError:
                parse_dates = False

            df = pd.read_csv(
                path, sep=self.sep, encoding="utf-8", low_memory=False,
                parse_dates=parse_dates if parse_dates else None, date_parser=date_parser if parse_dates else None
            )

            col_in_df = check_col_in_df(df, col)
            if col_in_df:
                self.dict_dataframes[file] = df
            else:
                raise KeyError(f"{file} does not contain {col}")

    @timeit
    def check_cycles(self):
        """
        Check if tables and their relations are contained cycles. Id bad for features generating.
        """
        dict_tables_for_vertices = {list(self.tables.keys())[i]: i for i in range(len(list(self.tables.keys())))}
        list_edges = [
            (dict_tables_for_vertices[rel[0][0]], dict_tables_for_vertices[rel[1][0]]) for rel in self.relations
        ]
        graph = nx.Graph(list_edges)

        try:
            cycle = nx.find_cycle(graph)
        except nx.exception.NetworkXNoCycle:
            cycle = None

        if cycle is not None:
            dict_vertices_for_table = {v: k for k, v in dict_tables_for_vertices.items()}
            cycle_to_table = [(dict_vertices_for_table[e[0]], dict_vertices_for_table[e[1]]) for e in cycle]
            raise ValueError(f"Check relations, cycle is founded: {cycle_to_table}")

    @timeit
    def create_entityset(self):
        """
        Make EntitySet object from DataFrames.
        """
        self.entities = ft.EntitySet(id='entities')
        for table, col_index in self.tables.items():
            try:
                time_index = self.time_indecies[table]  # тип должен быть str, не list
            except KeyError:
                time_index = False
            if col_index:
                self.entities = self.entities.entity_from_dataframe(
                    entity_id=table,
                    dataframe=self.dict_dataframes[table],  # dataframe object
                    index=col_index,  # unique index
                    # variable_types=app_types,  # defined specific data types (if needed)
                    time_index=time_index if time_index else None
                )
            else:
                index_name = table[:table.rfind('.csv')] + '_index'
                self.drop_contains.append(index_name)
                self.entities = self.entities.entity_from_dataframe(
                    entity_id=table,
                    dataframe=self.dict_dataframes[table],
                    make_index=True,  # need to create unique index from scratch
                    index=index_name,  # name it
                    time_index=time_index if time_index else None
                )

    @timeit
    def create_relations(self):
        """
        Add relations to EntitySet object.
        """
        list_relations_in_ft_format = list()
        for rel in self.relations:

            table_0 = rel[0][0]
            col_0 = rel[0][1]
            table_1 = rel[1][0]
            col_1 = rel[1][1]

            if check_col_in_df(self.dict_dataframes[table_0], col_0) and \
                    check_col_in_df(self.dict_dataframes[table_1], col_1):

                rel_ = ft.Relationship(
                    self.entities[table_0][col_0],
                    self.entities[table_1][col_1]
                )

                list_relations_in_ft_format.append(rel_)

            else:
                raise ValueError(f"Check key column in {table_0} or {table_1}")

        self.entities = self.entities.add_relationships(list_relations_in_ft_format)
        logging.info(f"Entityset is created \n {self.entities}")

    @timeit
    def plot_entityset(self):
        """
        Make jpg file with tables schema.
        """
        try:
            self.entities.plot('plot_entityset.jpg')
        except BaseException as ex:
            logging.error(f"Can't plot entityset \n {ex}")

    @timeit
    def feature_matrix_base_parallel(self):
        """
        Make future matrix (DataFrame object) from EntitySet object and save result in csv file.
        """
        logging.info(f"Start {self.n_jobs} thread(s)" if self.n_jobs != -1 else f"Start {os.cpu_count()} thread(s)")
        self.feature_matrix, self.feature_names = ft.dfs(
            entityset=self.entities,
            target_entity=self.main_table,
            # ignore_variables={self.main_table: [self.target]},
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            max_depth=self.depth,
            verbose=True,
            n_jobs=self.n_jobs,
            chunk_size=self.chunk_size,
            drop_contains=self.drop_contains,
            max_features=self.max_features,
        )
        logging.info(f"Feature matrix final size: {self.feature_matrix.shape}")
        gc.collect()
        save_dataframe_to_csv(self.feature_matrix, os.path.join(self.path, self.output_file_name), self.sep)
