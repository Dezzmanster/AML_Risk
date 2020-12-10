import time 
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing
import multiprocessing as mp
import pickle
import logging.config
import logging
import warnings
warnings.filterwarnings('ignore')
logging.config.fileConfig(fname='logger.ini', defaults={'logfilename': 'logfile.log'})

def save_to_csv(X, rest_columns=None, path=None):
    '''
    X: pd.DataFrame. The main table.
    rest_columns: pd.Series. The rest columns to concat with X before saving.
    path: string. The path where to save the table.
    '''
    if path is None:
        path = 'trial_{}.csv'.format(time.strftime("%m%d%Y-%H:%M"))
    if rest_columns is not None:
        col_XX_rest= list(X.columns) + list(pd.DataFrame(rest_columns).columns)
        pd.DataFrame(np.hstack([X.to_numpy(), rest_columns.to_numpy().reshape(len(rest_columns),1)]), columns=col_XX_rest).to_csv(path, index=False)
        print('\n Successfully saved to {}'.format(path))
        logging.info(f"Successfully saved to {path}")
    else:
        X.to_csv(path, index=False)
        print('\n Successfully saved to {}'.format(path))
        logging.info(f"Successfully saved to {path}")
        
def reduce_mem_usage(X):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = X.memory_usage().sum() / 1024 ** 2
    print()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    logging.info(f"Memory usage of dataframe is {start_mem} MB'")
    for col in X.columns:
        col_type = X[col].dtype
        if not any(str(col_type) == t for t in ['category', 'object', 'datetime64[ns]', 'datetime64[ns, tz]']):
            c_min = X[col].min()
            c_max = X[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    X[col] = X[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    X[col] = X[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    X[col] = X[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    X[col] = X[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    X[col] = X[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    X[col] = X[col].astype(np.float32)
                else:
                    X[col] = X[col].astype(np.float64)
    end_mem = X.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Memory usage decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    logging.info(f"Memory usage after optimization is: {end_mem} MB")
    logging.info(f"Memory usage decreased by {100 * (start_mem - end_mem) / start_mem}%")
    return X

class EmptyElim(object):
    '''
    n_jobs: int, default = None. The number of jobs to run in parallel.
        None - means 1
        -1 - means using all processors
    chunks: int, default = None. Number of features sent to processor per time.
        None - means number of features/number of cpu
    '''
    def __init__(self, n_jobs=None, chunks=None):
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.chunks = chunks
        logging.info(f"Object {self} is created")

    def detect_col_(self, X):
        for column in X.columns:
            if len(X[column].unique()) < 2:
                self.col_names[column] = list(X[column].unique())
        return self.col_names

    def drop_col_(self, X):
        columns = [i for i in list(self.col_names.keys()) if i in list(X.columns)]
        X.drop(columns=columns, inplace=True)
        return X

    def fit(self, X):
        '''
        Create a dictionary of names of columns (self.colnames) to drop.
        '''
        self.col_names = {}
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X / self.n_jobs) <= 1:
                self.n_jobs -= 1
            self.chunks = int(n_columns_X / self.n_jobs)
        return_ = mp.Pool(processes=self.n_jobs).map(self.detect_col_,
                                                     [X[columns_X[start: start + self.chunks]] for start in
                                                      range(0, n_columns_X, self.chunks)]
                                                     )
        for r in return_:
            self.col_names.update(r)
        print('\n columns to drop:', self.col_names)
        logging.info(f"Dictionary of names of columns to drop has been created: {self.col_names}")

    def transform(self, X, file_path=None):
        '''
        Drops self.col_names that were initialized by self.fit function.
        '''
        logging.info(f"Initial X shape: {X.shape}")
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X / self.n_jobs) <= 1:
                self.n_jobs -= 1
            self.chunks = int(n_columns_X / self.n_jobs)
        X = pd.concat(mp.Pool(processes=self.n_jobs).map(self.drop_col_,
                                                         [X[columns_X[start: start + self.chunks]] for start in
                                                          range(0, n_columns_X, self.chunks)]
                                                         ), axis=1)
        logging.info(f"Columns were dropped, X new shape: {X.shape}")
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X

    def fit_transform(self, X, file_path=None):
        '''
        Find self.col_names and drops them.
        '''
        self.fit(X)
        X = self.transform(X)
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X

class FEncoding(object):
    def __init__(self, n_jobs=1, chunks=None, rest_col_names=[]):
        self.rest_col_names = rest_col_names
        self.categor_types = ['category', 'object', 'bool', 'int8', 'int16', 'int32', 'int64']
        self.numer_types = ['float', 'float8', 'float16', 'float32', 'float64']
        self.time_types = ['datetime64[ns]', 'datetime64[ns, tz]']
        if n_jobs == None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.chunks = chunks
        self.categor_columns_keep, self.numer_columns_keep = [], []
        logging.info(f"Object {self} is created")

    def date_replace_(self, X):
        def pars_date(x):
            fmts = ('%Y', '%b %d, %Y', '%b %d, %Y', '%B %d, %Y', '%B %d %Y', '%m/%d/%Y', '%m/%d/%y', '%b %Y', '%B%Y',
                    '%b %d,%Y',
                    '%d.%m.%Y', '%Y.%m.%d', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S')
            t = True
            if str(x.dtype) == 'object':
                for fmt in fmts:
                    try:
                        return pd.Series([dt.datetime.strptime(str(x.iloc[i]), fmt) for i in range(len(x))]).apply(
                            lambda q: q.strftime('%m/%d/%Y')).astype('datetime64[ns]')
                        t = False
                        break
                    except ValueError:
                        pass
            if t and (len(str(x.iloc[0])) > 9) and (len(str(x.iloc[0])) <= 14):
                # TODO: better condition on string to identify that it is unix timestep
                try:
                    x = x.astype('float')
                    return pd.Series([dt.datetime.fromtimestamp(x.iloc[i]) for i in range(len(x))]).apply(
                        lambda q: q.strftime('%m/%d/%Y')).astype('datetime64[ns]')
                except ValueError:
                    pass

        f_columns_names = [x for x in list(X.columns) if x not in self.rest_col_names]
        for column in f_columns_names:
            x = pars_date(X[column])
            try:
                x.nunique()
                X[column] = x
            except AttributeError:
                pass
        return X

    def initialize_types_(self, X):
        # Sometimes categorical feature can be presented with a float type.
        # Let's check for that
        categor_columns, numer_columns, time_columns = [], [], []
        f_columns_names = [x for x in list(X.columns) if x not in self.rest_col_names]
        for column in f_columns_names:
            c_type = str(X[column].dtype)
            unique_values = list(X[column].value_counts().index)
            if any(c_type == t for t in self.numer_types):
                # unique_values = list(np.unique(X[column][~np.isnan(X[column])]))
                # if np.array([el.item().is_integer() for el in unique_values]).sum() == len(unique_values):
                if np.array([el.is_integer() for el in unique_values]).sum() == len(unique_values):
                    # print('\n {} has type {} and number of unique values: {}, will be considered as a categorical \n'.format(column, c_type, len(unique_values)))
                    logging.info(
                        f"{column} has type {c_type} and number of unique values: {len(unique_values)}, will be considered as a categorical")
                    categor_columns.append(column)
                else:
                    numer_columns.append(column)
            elif any(c_type == t for t in self.categor_types):
                if len(unique_values) >= self.n_unique_val_th:
                    numer_columns.append(column)
                else:
                    categor_columns.append(column)
            elif any(c_type == t for t in self.time_types):
                time_columns.append(column)
        return categor_columns, numer_columns, time_columns

    def bucket_numerical_(self, X):
        # TODO: specify or introduce a criterion which columns to bake
        # K-bins discretization based on quantization
        def get_input_keypoints(f_data, n_kps):
            while len(np.quantile(f_data, np.linspace(0.0, 1.0, num=n_kps))) != len(
                    np.unique(np.quantile(f_data, np.linspace(0.0, 1.0, num=n_kps)))):
                n_kps -= 1
            return np.quantile(f_data, np.linspace(0.0, 1.0, num=n_kps))

        if self.columns_to_buck == 'all_numerical':
            self.columns_to_buck = self.numer_columns
        if type(self.columns_to_buck) != list:
            raise ValueError('Identify list of columns_to_buck')

        f_columns_names = [x for x in list(X.columns) if x not in self.rest_col_names]
        for column in f_columns_names:
            if any(column == col for col in self.columns_to_buck):
                # print('\n {} bucketing ...'.format(column))
                logging.info(f"{column} bucketing ...")
                f_X = X[column].values.ravel()
                keypoints = get_input_keypoints(f_X, self.n_bins)
                if len(pd.Series(keypoints).value_counts()) <= 1:
                    # print('\n', column, 'has keypoints:', keypoints, ', and can not be bucketed.')
                    logging.info(f"{column} has keypoints: {keypoints},  and can not be bucketed")
                    pass
                else:
                    X[str(column) + '_bucketed'] = np.digitize(f_X, keypoints)
                    if self.drop_current:
                        X.drop(columns=[column], inplace=True)
            else:
                print('\n ', column, 'is not numerical!')
        return X

    def encode_categor_(self, X, method='OrdinalEncoder'):
        if self.method == 'OrdinalEncoder':
            mask = X.isnull().to_numpy()
            X_original = X
            X = X.astype(str)
            enc = preprocessing.OrdinalEncoder(dtype='int')
            X = pd.DataFrame(enc.fit_transform(X), columns=X.columns)
            X = X.mask(mask, X_original)
        if self.method == 'OneHotEncoder':
            X = X.astype(object)
            X = pd.get_dummies(X)
        return X

    def encode_time_(self, X):
        f_columns_names = [x for x in list(X.columns) if x not in self.rest_col_names]
        for column in X.columns:
            if any(str(X[column].dtype) == t for t in self.time_types):
                X[str(column) + '_year'] = X[column].dt.year
                X[str(column) + '_month'] = X[column].dt.month
                X[str(column) + '_day'] = X[column].dt.day
                print('\n {} was encoded from date'.format(column))
                if self.drop_current:
                    X.drop(columns=[column], inplace=True)
        return X

    def initialize_types(self, X, n_unique_val_th=50, categor_columns_keep=[], numer_columns_keep=[],
                         return_dtype=False, file_name=None):
        # For other modules
        global n_unique_val_th_, categor_columns_keep_, numer_columns_keep_
        n_unique_val_th_, categor_columns_keep_, numer_columns_keep_ = n_unique_val_th, categor_columns_keep, numer_columns_keep
        
        self.n_unique_val_th = n_unique_val_th
        self.categor_columns_keep, self.numer_columns_keep = categor_columns_keep, numer_columns_keep
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X / self.n_jobs) <= 1:
                self.n_jobs -= 1
            self.chunks = int(n_columns_X / self.n_jobs)
        return_ = mp.Pool(processes=self.n_jobs).map(self.initialize_types_,
                                                     [X[columns_X[start: start + self.chunks]] for start in
                                                      range(0, n_columns_X, self.chunks)]
                                                     )
        self.categor_columns, self.numer_columns, self.time_columns = [], [], []
        for i in range(len(return_)):
            categor_columns, numer_columns, time_columns = return_[i]
            self.categor_columns += categor_columns
            self.numer_columns += numer_columns
            self.time_columns += time_columns
        self.categor_columns += categor_columns_keep
        self.numer_columns += numer_columns_keep
        self.categor_columns = list(set(self.categor_columns))
        self.numer_columns = list(set(self.numer_columns))
        self.time_columns = list(set(self.time_columns))
        if len(self.numer_columns_keep) != 0:
            for f_n in self.numer_columns_keep:
                if any(f_n == t for t in self.categor_columns):
                    self.categor_columns.remove(f_n)
        if len(self.categor_columns_keep) != 0:
            for f_c in self.categor_columns_keep:
                if any(f_c == t for t in self.numer_columns):
                    self.numer_columns.remove(f_c)
        logging.info(f"Types have been initialized.")
        out_dict = {'categor_columns': self.categor_columns,
                    'numer_columns': self.numer_columns,
                    'time_columns': self.time_columns}
        if return_dtype:
            out_dict.update(
                {'categor_columns_dtypes': [str(X[self.categor_columns].dtypes.values[i]) for i in
                                            range(len(self.categor_columns))],
                 'numer_columns_dtypes': [str(X[self.numer_columns].dtypes.values[i]) for i in
                                          range(len(self.numer_columns))],
                 'time_columns_dtypes': [str(X[self.time_columns].dtypes.values[i]) for i in
                                         range(len(self.time_columns))], })

        if file_name != None:
            output = open(file_name, 'wb')
            pickle.dump(out_dict, output)
            output.close()
        return out_dict

    def bucket_numerical(self, X,
                         n_bins=5, columns_to_buck='all_numerical',
                         drop_current=False, file_path=None):
        self.n_bins = n_bins
        self.columns_to_buck = columns_to_buck
        self.drop_current = drop_current
        try:
            self.time_columns
        except AttributeError:
            try:
                self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
            except NameError:
                self.initialize_types(X)
        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]

        n_columns_X = len(numer_columns)
        if n_columns_X != 0:
            if self.chunks == None:
                while int(n_columns_X / self.n_jobs) <= 1:
                    self.n_jobs -= 1
                self.chunks = int(n_columns_X / self.n_jobs)
            X_num = pd.concat(mp.Pool(processes=self.n_jobs).map(self.bucket_numerical_,
                                                                 [X[numer_columns[start: start + self.chunks]] for start
                                                                  in range(0, n_columns_X, self.chunks)]
                                                                 ), axis=1)
            rest_col = categor_columns + time_columns
            if len(rest_col) != 0:
                X = pd.concat([X[rest_col], X_num], axis=1)
            else:
                X = X_num
            logging.info(f"Columns {numer_columns} have been bucketed.")
        else:
            print('\n No numerical columns in the dataset')
            logging.info(f"No numerical columns in the dataset")

        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X

    def encode_categor(self, X, method='OrdinalEncoder', file_path=None):
        self.method = method
        try:
            self.time_columns
        except AttributeError:
            try:
                self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
            except NameError:
                self.initialize_types(X)
        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]
        n_columns_X = len(categor_columns)
        if n_columns_X != 0:
            if self.chunks == None:
                while int(n_columns_X / self.n_jobs) <= 1:
                    self.n_jobs -= 1
                self.chunks = int(len(categor_columns) / self.n_jobs)
            X_cat = pd.concat(mp.Pool(processes=self.n_jobs).map(self.encode_categor_,
                                                                 [X[categor_columns[start: start + self.chunks]] for
                                                                  start in range(0, len(categor_columns), self.chunks)]
                                                                 ), axis=1)
            rest_col = numer_columns + time_columns
            if len(rest_col) != 0:
                X = pd.concat([X[rest_col], X_cat], axis=1)
            else:
                X = X_cat
            logging.info(f"Columns {categor_columns} have been encoded.")
        else:
            print('\n No categorical columns in the dataset')
            logging.info(f"No categorical columns in the dataset")

        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X

    def encode_time(self, X, drop_current=False, file_path=None):
        self.drop_current = drop_current
        try:
            self.time_columns
        except AttributeError:
            try:
                self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
            except NameError:
                self.initialize_types(X)
        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]
        n_columns_X = len(time_columns)
        if n_columns_X != 0:
            if self.chunks == None:
                while int(n_columns_X / self.n_jobs) <= 1:
                    self.n_jobs -= 1
                self.chunks = int(n_columns_X / self.n_jobs)
            X_time = pd.concat(mp.Pool(processes=self.n_jobs).map(self.encode_time_,
                                                                  [X[time_columns[start: start + self.chunks]] for start
                                                                   in range(0, n_columns_X, self.chunks)]
                                                                  ), axis=1)
            rest_col = numer_columns + categor_columns
            if len(rest_col) != 0:
                X = pd.concat([X[rest_col], X_time], axis=1)
            else:
                X = X_time
            logging.info(f"Columns {time_columns} have been encoded.")
        else:
            print('\n No time columns in the dataset')
            logging.info(f"No time columns in the dataset")
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X

    def date_replace(self, X, file_path=None):
        try:
            self.time_columns
        except AttributeError:
            try:
                self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
            except NameError:
                self.initialize_types(X)
        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]
        n_columns_X = len(time_columns)
        if n_columns_X != 0:
            if self.chunks == None:
                while int(n_columns_X / self.n_jobs) <= 1:
                    self.n_jobs -= 1
                self.chunks = int(n_columns_X / self.n_jobs)
            X = pd.concat(mp.Pool(processes=self.n_jobs).map(self.date_replace_,
                                                             [X[columns_X[start: start + self.chunks]] for start in
                                                              range(0, n_columns_X, self.chunks)]
                                                             ), axis=1)
            print('\n time_columns:', self.time_columns)
            logging.info(f"Columns {self.time_columns} have been detected as date and encoded as yy-mm-dd.")
        else:
            print('\n No time columns in the dataset')
            logging.info(f"No time columns in the dataset")
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X


class FImputation(FEncoding):
    '''
      Dealing with missing values:
      We will use simple techniques with regards to the model that we use.
      For tree-based models, nana will be filled in with max values (or zeros)
      For regression with means and medians for numerical and categorical types respectively.
    '''

    def __init__(self, model_type, fill_with_value=None,
                 n_jobs=None, chunks=None):
        super(FImputation, self).__init__()

        self.model_type = model_type
        self.fill_with_value = fill_with_value
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.chunks = chunks
        logging.info(f"Object {self} is created")

    def impute_(self, X):
        if self.model_type == 'tree-based':
            if self.fill_with_value == 'zeros':
                X.fillna(0, inplace=True)
                return X
            elif self.fill_with_value == 'extreme_values':
                for column in X.columns:
                    X[column].fillna(X[column][abs(X[column]) == abs(X[column]).max()].values[0], inplace=True)
                return X
            else:
                raise ValueError(f"Identify fill_with_value parameter")

        if self.model_type == 'regression-based':
            categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
            numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
            for column in X.columns:
                unique_values = np.unique(X[column][~pd.isnull(X[column])])
                if any(column == t for t in categor_columns):
                    X[column].fillna(int(np.median(unique_values)), inplace=True)
                if any(column == t for t in numer_columns):
                    X[column].fillna(np.mean(unique_values), inplace=True)
            return X

    def impute(self, X, file_path=None):
        try:
            self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
        except NameError:
            self.initialize_types(X)
        X = self.encode_categor(X, method='OrdinalEncoder')
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X / self.n_jobs) <= 1:
                self.n_jobs -= 1
            self.chunks = int(n_columns_X / self.n_jobs)
        X = pd.concat(mp.Pool(processes=self.n_jobs).map(self.impute_,
                                                         [X[columns_X[start: start + self.chunks]] for start in
                                                          range(0, n_columns_X, self.chunks)]
                                                         ), axis=1)
        logging.info(f"Successfully imputed")
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X 

class OutlDetect(FEncoding):
    '''
    outliers_detection_technique: {'iqr_proximity_rule', 'gaussian_approximation', 'quantiles'}, default = 'iqr_proximity_rule'
        'iqr_proximity_rule' - the boundaries are determined using IQR proximity rules
        'gaussian_approximation' - sets the boundaries with values according to the mean and standard deviation
        'quantiles' - the boundaries are determined using the quantiles, through which you can specify any percentage you want
         Source: https://heartbeat.fritz.ai/hands-on-with-feature-engineering-techniques-dealing-with-outliers-fcc9f57cb63b

        Note: Categorical Outliers Donâ€™t Exist
        Source: https://medium.com/owl-analytics/categorical-outliers-dont-exist-8f4e82070cb2

    n_jobs: int, default = None. The number of jobs to run in parallel.
        None - means 1
        -1 - means using all processors

    chunks: int, default = None. Number of features sent to processor per time.
        None - means number of features/number of cpu
    '''

    def __init__(self, outliers_detection_technique='iqr_proximity_rule', n_jobs=None,
                 chunks=None):
        super(OutlDetect, self).__init__()
        self.outliers_detection_technique = outliers_detection_technique
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.chunks = chunks
        logging.info(f"Object {self} is created")

    def iqr_proximity_rule(self, X):
        for column in X.columns:
            x = X[column]
            IQR = x.quantile(0.75) - x.quantile(0.25)
            lower = x.quantile(0.25) - (IQR * 1.5)
            upper = x.quantile(0.75) + (IQR * 1.5)
            self.col_outl_info[column] = (lower, upper)
        return self.col_outl_info

    def replace(self, X):
        for column in X.columns:
            x = X[column]
            lower, upper = self.col_outl_info[column]
            X[column] = np.where(x > upper, upper, np.where(x < lower, lower, x))
        return X

    def gaussian_approximation(self, X):
        # Gaussian approximation
        for column in X.columns:
            x = X[column]
            lower = x.mean() - 3 * x.std()
            upper = x.mean() + 3 * x.std()
            self.col_outl_info[column] = (lower, upper)
        return self.col_outl_info

    def quantiles(self, X):
        # Using quantiles
        for column in X.columns:
            x = X[column]
            lower = x.quantile(0.10)
            upper = x.quantile(0.90)
            self.col_outl_info[column] = (lower, upper)
        return self.col_outl_info

    def fit(self, X):
        '''
        Collect information regarding self.col_outl_info - lower and upper bounds to clip outliers.
        '''
        try:
            self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
        except NameError:
            self.initialize_types(X)
        self.categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        self.numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        self.time_columns = [i for i in self.time_columns if i in list(X.columns)]

        n_columns_X = len(self.numer_columns)
        if n_columns_X != 0:
            if self.chunks is None:
                while int(n_columns_X / self.n_jobs) <= 1:
                    self.n_jobs -= 1
                self.chunks = int(n_columns_X / self.n_jobs)

            if self.outliers_detection_technique == 'iqr_proximity_rule':
                f = self.iqr_proximity_rule
            elif self.outliers_detection_technique == 'gaussian_approximation':
                f = self.gaussian_approximation
            elif self.outliers_detection_technique == 'quantiles':
                f = self.quantiles

            self.col_outl_info = {}
            return_ = mp.Pool(processes=self.n_jobs).map(f,
                                                         [X[self.numer_columns[start: start + self.chunks]] for start in
                                                          range(0, n_columns_X, self.chunks)]
                                                         )
            for r in return_:
                self.col_outl_info.update(r)
            print('\n col_outl_info (upper, lower) bounds:', self.col_outl_info)
            logging.info(
                f"{self.outliers_detection_technique}, col_outl_info (upper, lower) bounds:{self.col_outl_info}")
        else:
            print('\n No numerical columns in the dataset')
            logging.info(f"No numerical columns in the dataset")

    def transform(self, X, file_path=None):
        '''
        Clip ouliers by using the dict of lower and upper bounds (self.col_outl_info).
        '''
        n_columns_X = len(self.numer_columns)
        if n_columns_X != 0:
            if self.chunks == None:
                while int(n_columns_X / self.n_jobs) <= 1:
                    self.n_jobs -= 1
                self.chunks = int(n_columns_X / self.n_jobs)
            X_num = pd.concat(mp.Pool(processes=self.n_jobs).map(self.replace,
                                                                 [X[self.numer_columns[start: start + self.chunks]] for start
                                                                  in range(0, n_columns_X, self.chunks)]
                                                                 ), axis=1)
            rest_col = self.categor_columns + self.time_columns
            if len(rest_col) != 0:
                X = pd.concat([X[rest_col], X_num], axis=1)
            else:
                X = X_num
            logging.info(f"Successfully clipped")
        else:
            print('\n No numerical columns in the dataset')
            logging.info(f"No numerical columns in the dataset")
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X

    def fit_transform(self, X, file_path=None):
        '''
        1. Collect information regarding self.col_outl_info - lower and upper bounds to clip outliers.
        2. Clip ouliers by using the dict of lower and upper bounds (self.col_outl_info).
        '''
        self.fit(X)
        X = self.transform(X)
        if file_path is not None:
            X.to_csv(file_path, index=False)
        return X
