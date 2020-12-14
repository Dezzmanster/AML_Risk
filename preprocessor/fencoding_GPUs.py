import time
import numpy as np
import pandas as pd

# Standard Libraries
import os
import glob
import shutil
import nvidia_smi

# External Dependencies
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
import dask
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import dask.dataframe as dd
import dask.array as da
import rmm

import dask_ml
from dask_ml.preprocessing import OneHotEncoder

import pickle

print('Dask Version:', dask.__version__)
print('Dask cuDF Version:', dask_cudf.__version__)
print()

# NVTabular
import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io import Shuffle
from nvtabular.utils import device_mem_size

import logging.config
import logging
import warnings
warnings.filterwarnings('ignore')
#logging.config.fileConfig(fname='logger.ini', defaults={'logfilename': 'logfile.log'})

def set_cluster_client(n_gpus=-1, device_spill_frac=0.8):
        # TODO: Check for any solution. If user calls this function, for the second call the correct recreation will fail. 
        # New cluster can be created after 'kernel restart' procedure.
        '''
        device_spill_frac: Spill GPU-Worker memory to host at this limit. Reduce if spilling fails to prevent device memory errors.
        '''
        if os.path.isdir("dask-worker-space"):
            shutil.rmtree('dask-worker-space', ignore_errors=True)       
        # Deploy a Single-Machine Multi-GPU Cluster
        if n_gpus == -1:
            nvidia_smi.nvmlInit()
            n_gpus_avail = nvidia_smi.nvmlDeviceGetCount()
            print('\n n_gpus_avail: {}'.format(n_gpus_avail))
            n_gpus = n_gpus_avail
        # Delect devices to place workers
        visible_devices = [i for i in list(range(n_gpus))]
        visible_devices = str(visible_devices)[1:-1]
        #print('visible_devices: {}'.format(visible_devices))
            
        #TODO: how to reinitialzed cluster
        cluster = LocalCUDACluster(
            protocol = "tcp", # "tcp" or "ucx"
            CUDA_VISIBLE_DEVICES = visible_devices,
            device_memory_limit = device_spill_frac * device_mem_size(kind="total"),
        )       
        try: 
            # Create the distributed client
            client = Client(cluster)
            display(client)
            print('\n Dashboard avail: http://localhost:8888/proxy/8787/status')
            # Initialize RMM pool on ALL workers
            def _rmm_pool():
                rmm.reinitialize(
                    pool_allocator=True,
                    initial_pool_size=None, # Use default size
                )         
            client.run(_rmm_pool)
            return client
        except MemoryError:
            print('\n The client is already initialized')
            

class FEncoding_advanced(object):   
    def __init__(self, client, rest_col_names=[]):        
        self.n_gpus = len(client.nthreads())
        self.client = client
        self.rest_col_names=rest_col_names
        self.output_path="./parquet_data_tmp"
        self.categor_types = ['category', 'object', 'bool', 'int8', 'int16', 'int32', 'int64', 'int8']
        self.numer_types = ['float', 'float8', 'float16', 'float32', 'float64']
        self.time_types = ['datetime64[ns]', 'datetime64[ns, tz]']
        logging.info(f"Object {self} is created")
        
    def elim_empty_columns(self, X, file_path=None):
        # GPU version
        if type(X) == pd.core.frame.DataFrame:
            X = dd.from_pandas(X, npartitions=self.n_gpus)
        cols_to_drop = {}
        for column in X.columns:
            unique_values = X[column].unique().compute().values
            if len(unique_values) < 2:
                cols_to_drop = list(cols_to_drop.keys())
        print('\n columns to drop:', cols_to_drop)
        X = X.drop(cols_to_drop, axis=1).compute()
        if file_path is not None:
            X.to_csv(file_path, index=False)
        logging.info(f"Columns were dropped, X new shape: {X.shape}")
        return X
    
    def initialize_types(self, X, n_unique_val_th=50, categor_columns_keep=[], numer_columns_keep=[],
                         return_dtype=False, file_name=None):
        # For other modules
        global n_unique_val_th_, categor_columns_keep_, numer_columns_keep_
        n_unique_val_th_, categor_columns_keep_, numer_columns_keep_ = n_unique_val_th, categor_columns_keep, numer_columns_keep
        
        # GPU version
        if type(X) == pd.core.frame.DataFrame:
            X = dd.from_pandas(X, npartitions=self.n_gpus)
            
        self.categor_columns, self.numer_columns, self.time_columns = [], [], []
        f_columns_names =[x for x in list(X.columns)  if x not in self.rest_col_names]
        for column in f_columns_names:
            c_type = str(X[column].dtype)
            unique_values = list(X[column].value_counts().index.compute())
            if any(c_type == t for t in self.numer_types):
                if cp.array([el.is_integer() for el in unique_values]).sum() == len(unique_values):
                    logging.info(f"{column} has type {c_type} and number of unique values: {len(unique_values)}, will be considered as categorical")
                    self.categor_columns.append(column)
                else:
                    self.numer_columns.append(column)
            elif any(c_type == t for t in self.categor_types):
                if len(unique_values) >= n_unique_val_th:
                    self.numer_columns.append(column)
                    logging.info(f"{column} has type {c_type} and number of unique values: {len(unique_values)}, will be considered as numerical")
                else:
                    self.categor_columns.append(column)
            elif any(c_type == t for t in self.time_types):
                self.time_columns.append(column)                             
        out_dict =  {'categor_columns': self.categor_columns,
                'numer_columns': self.numer_columns,
                'time_columns': self.time_columns,                    
         }
        self.categor_columns += categor_columns_keep
        self.numer_columns += numer_columns_keep
        self.categor_columns = list(set(self.categor_columns))
        self.numer_columns = list(set(self.numer_columns))
        self.time_columns = list(set(self.time_columns))
        if len(numer_columns_keep) != 0:
            for f_n in numer_columns_keep:
                if any(f_n == t for t in self.categor_columns):
                    self.categor_columns.remove(f_n)
        if len(categor_columns_keep) != 0:
            for f_c in categor_columns_keep:
                if any(f_c == t for t in self.numer_columns):
                    self.numer_columns.remove(f_c)
        logging.info(f"Types have been initialized.")
        if return_dtype:
            out_dict.update(
                {'categor_columns_dtypes': [str(X[self.categor_columns].dtypes.values[i]) for i in range(len(self.categor_columns))],
                 'numer_columns_dtypes': [str(X[self.numer_columns].dtypes.values[i]) for i in range(len(self.numer_columns))],
                 'time_columns_dtypes': [str(X[self.time_columns].dtypes.values[i]) for i in range(len(self.time_columns))],                    
             })            
        if file_name != None:
            output = open(file_name, 'wb')
            pickle.dump(out_dict, output)
            output.close()
        return out_dict
    
    def outldetect(self, outliers_detection_technique, X_num):
        # GPU version
        if outliers_detection_technique == 'iqr_proximity_rule':  
            IQR = (X_num.quantile(0.75).sub(X_num.quantile(0.25)))
            lower = X_num.quantile(0.25).sub(IQR*1.5)
            upper = X_num.quantile(0.75).sub(IQR*1.5)            
        if outliers_detection_technique == 'gaussian_approximation':
            lower = X_num.mean().sub(3 * X_num.std())
            upper = X_num.mean().add(3 * X_num.std())        
        if outliers_detection_technique == 'quantiles':
            lower = X_num.quantile(0.10)
            upper = X_num.quantile(0.90)        
        return list(lower.compute()), list(upper.compute()) 
    
    def extrvalsdetect(self, X, f_type):
        # GPU version
        extrim_values = {}
        columns_names = list(X.columns)
        if f_type == 'numer_columns':
            extrvalar = cp.array(X.where(X.abs() == X.abs().max()).values.compute()).T
            for i in range(len(columns_names)):
                extrim_values[columns_names[i]] = cp.unique(extrvalar[i,:])[0].item()            
        if f_type == 'categor_columns':
            for column in columns_names:
                if len(X[column].unique().compute().values) <= 2:
                    pass
                else:
                    extrim_values[column] = X[column].value_counts().compute().index[-1]  
        return extrim_values
    
    def processing(self, X_pd, y_names = [],
                   encode_categor_type = None, 
                   #'categorify', 'onehotencoding',
                   outliers_detection_technique = None,
                   #'iqr_proximity_rule', 'gaussian_approximation','quantiles'
                   fill_with_value = None,
                   #'extreme_values', 'zeros','mean-median'
                   targetencoding = False,
                   file_path=None,
                  ):
        X = dd.from_pandas(X_pd, npartitions=self.n_gpus)
        X=X.replace(np.nan, None)        
        try:
            self.time_columns
        except AttributeError:
            try:
                self.initialize_types(X, n_unique_val_th = n_unique_val_th_,
                              categor_columns_keep = categor_columns_keep_, numer_columns_keep = numer_columns_keep_)
            except NameError:
                self.initialize_types(X)
        
        workflow = nvt.Workflow(cat_names=self.categor_columns, 
                        cont_names=self.numer_columns,
                        label_name=y_names,
                        client=self.client)
        # Operators: https://nvidia.github.io/NVTabular/main/api/ops/index.html      
        # Categorify https://nvidia.github.io/NVTabular/main/api/ops/categorify.html
        if encode_categor_type == 'categorify':
            if len(self.categor_columns) != 0:
                workflow.add_preprocess(
                    ops.Categorify(columns = self.categor_columns, out_path='./'))
        
        if encode_categor_type == 'onehotencoding':
            #OneHotEncoder().get_feature_names(input_features=<list of features encoded>) does not work
            #lengths=True - chunk sizes can be computed
            for column in self.categor_columns:
                #X[column] = X[column].astype(str)
                X_cat_encoded = OneHotEncoder().fit_transform(X[column].to_dask_array(lengths=True).reshape(-1,1))
                uvs = X[column].unique().compute().values
                X = X.drop([column], axis=1)
                X_cat_encoded = dd.from_array(X_cat_encoded.compute().todense())
                X_cat_encoded.columns = [column + '_{}'.format(uv) for uv in uvs]
                X = dd.concat([X,X_cat_encoded], axis=1)
                X = X.repartition(npartitions=2)    
            for column in X.columns:
                if any(str(column)[-4:] == t for t in ['_nan', 'None']): # What else?
                    X = X.drop([column], axis=1)
                
            self.initialize_types(X)
            print('Retyping:', self.initialize_types(X))
            # Reinitialize workflow
            workflow = nvt.Workflow(cat_names=self.categor_columns, 
                cont_names=self.numer_columns,
                label_name=y_names,
                client=self.client)

        # OutlDetect https://nvidia.github.io/NVTabular/main/api/ops/clip.html
        if (len(self.numer_columns) != 0) and (outliers_detection_technique != None):
            lower, upper = self.outldetect(outliers_detection_technique, X[self.numer_columns])
            for i in range(len(self.numer_columns)):
                logging.info(f'column: {self.numer_columns[i]}, lower: {lower[i]}, upper: {upper[i]}')
                print(f'column: {self.numer_columns[i]}, lower: {lower[i]}, upper: {upper[i]}')
                workflow.add_preprocess(
                    ops.Clip(min_value=lower[i], max_value=upper[i], columns=[self.numer_columns[i]])
                )
                
        # FillMissing https://nvidia.github.io/NVTabular/main/api/ops/fillmissing.html
        if fill_with_value == 'zeros':
            workflow.add_preprocess(
                ops.FillMissing(fill_val=0, columns=self.categor_columns + self.numer_columns)) 
            
        if fill_with_value == 'extreme_values':
            extrim_values = {}
            if len(self.numer_columns) != 0:
                extrim_values.update(self.extrvalsdetect(X[self.numer_columns], 'numer_columns'))
                
            if len(self.categor_columns) != 0:
                extrim_values.update(self.extrvalsdetect(X[self.categor_columns], 'categor_columns'))
            logging.info(f'extrim_values: {extrim_values}')
            
            output = open('extrim_values', 'wb')
            pickle.dump(extrim_values, output)
            output.close()
            
            for fill_val, column in zip(list(extrim_values.values()), list(extrim_values.keys())):
                workflow.add_preprocess(
                    ops.FillMissing(fill_val=fill_val, columns=[column]))
                
        if fill_with_value == 'mean-median':
            if len(self.categor_columns) != 0:              
                workflow.add_preprocess(
                        ops.FillMedian(columns=self.categor_columns, preprocessing=True, replace=True)                    ) 
            if len(self.numer_columns) != 0:
                means = list(dd.from_pandas(X[self.numer_columns], 
                                            npartitions=self.n_gpus).mean().compute().values)
                for fill_val, column in zip(means, self.numer_columns):
                    workflow.add_preprocess(
                        ops.FillMissing(fill_val=fill_val, columns=[column]))
                    
        if targetencoding:
            #https://nvidia.github.io/NVTabular/main/api/ops/targetencoding.html
            if len(self.y_names) != 0:
                if len(self.cat_groups) == 0: 
                    print('\n Target encoding will be applied to all categorical columns')
                    workflow.add_preprocess(
                        ops.TargetEncoding(cat_groups = self.categor_columns,
                                            cont_target = self.y_names))
                else:
                    workflow.add_preprocess(
                        ops.TargetEncoding(cat_groups = self.cat_groups,
                                            cont_target = self.y_names))                                 
        #-----------------------------------------------------------------------------------------           
        workflow.finalize()
        dataset = nvt.Dataset(X)

        tmp_output_path="./parquet_data_tmp"
        workflow.apply(
             dataset,
             output_format="parquet",
             output_path=tmp_output_path,
             shuffle=Shuffle.PER_WORKER,  # Shuffle algorithm
             out_files_per_proc=1, # Number of output files per worker
             )
        files = glob.glob(tmp_output_path + "/*.parquet")
        X_final = cudf.read_parquet(files[0])
        for i in range(1, len(files)):    
            X_final = X_final.append(cudf.read_parquet(files[i]))      
        # Delete temporary files
        shutil.rmtree(tmp_output_path, ignore_errors=True)
#         if len(self.rest_col_names) != 0:
#             print(1)
#             X_final = pd.concat([X_final.to_pandas(), X_pd[self.rest_col_names]], axis=1)
        if file_path is not None:
            X_final.to_csv(file_path, index=False)
        return X_final 
            