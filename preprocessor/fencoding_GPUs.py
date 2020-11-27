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
import rmm

import pickle

print('Dask Version:', dask.__version__)
print('Dask cuDF Version:', dask_cudf.__version__)
print()

# NVTabular
import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io import Shuffle
from nvtabular.utils import device_mem_size

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
            print('TODO: memory reallocation')
            

class FEncoding_advanced(object):   
    def __init__(self, client, rest_col_names=[], y_names=[], filename=None):        
        self.filename = filename
        self.rest_col_names = rest_col_names
        self.y_names = y_names
        self.n_gpus = len(client.nthreads())
        self.client = client
        self.output_path="./parquet_data_tmp"
        self.categor_types = ['category', 'object', 'bool', 'int32', 'int64', 'int8']
        self.numer_types = ['float', 'float32', 'float64']
        self.time_types = ['datetime64[ns]', 'datetime64[ns, tz]'] 
        # What else? https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        # TODO: check if there are any other time types
        
    def elim_empty_columns(self, X, save_to_csv = False):
        # GPU version
        if type(X) == pd.core.frame.DataFrame:
            X = dd.from_pandas(X, npartitions=self.n_gpus)
        cols_to_drop = []
        for column in X.columns:
            if len(X[column].unique().compute().values) < 2:
                cols_to_drop.append(column)
        print('\n dropped columns:', cols_to_drop)
        X_final = X.drop(cols_to_drop, axis=1).compute()
        if save_to_csv:
            if self.filename is not None:
                X_final.to_csv('./data/' + self.filename, index=False)
            else:
                print('Identify filename when initializing the class!')
        return X_final
    
    def initialize_types(self, X, return_dtype=False, save_to_pkl = False, dict_name = 'out_dict.pkl'):
        # GPU version
        if type(X) == pd.core.frame.DataFrame:
            X = dd.from_pandas(X, npartitions=self.n_gpus)       
        self.categor_columns, self.numer_columns, self.time_columns = [], [], []
        # Sometimes categorical feature can be presented with a float type. Let's check for that
        f_columns_names =[x for x in list(X.columns)  if x not in self.rest_col_names + self.y_names]

        for column in f_columns_names:
            c_type = str(X[column].dtype) 

            if any(c_type == t for t in self.numer_types):

                uvs = cp.array(X[column].unique().compute())
                unique_values = list(uvs[~cp.isnan(uvs)])

                if cp.array([el.item().is_integer() for el in unique_values]).sum() == len(unique_values):
                    #print('\n {} has type {} and number of unique values: {}, will be considered as a categorical \n'.format(column, c_type, len(unique_values)))
                    #logging.info(f"{column} has type {c_type} and number of unique values: {len(unique_values)}, will be considered as a categorical")
                    self.categor_columns.append(column)
                else:
                    self.numer_columns.append(column)
            if any(c_type == t for t in self.categor_types):
                self.categor_columns.append(column)
            if any(c_type == t for t in self.time_types):
                self.time_columns.append(column)                             
        out_dict =  {'categor_columns': self.categor_columns,
                'numer_columns': self.numer_columns,
                'time_columns': self.time_columns,                    
         }
        if return_dtype:
            out_dict.update(
                {'categor_columns_dtypes': [str(X[self.categor_columns].dtypes.values[i]) for i in range(len(self.categor_columns))],
                 'numer_columns_dtypes': [str(X[self.numer_columns].dtypes.values[i]) for i in range(len(self.numer_columns))],
                 'time_columns_dtypes': [str(X[self.time_columns].dtypes.values[i]) for i in range(len(self.time_columns))],                    
             })            
        if save_to_pkl:
            output = open('./data/' + dict_name, 'wb')
            pickle.dump(out_dict, output)
            output.close()
        return out_dict
    
    def date_replace(self, X, save_to_csv = False):
        # GPU version
        return X
    
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
                extrim_values[column] = X[column].value_counts().compute().index[-1]  
        return extrim_values
    
    def processing(self, X, 
                   encode_categor_type = 'categorify',# 'onehotencoding',
                   outliers_detection_technique = 'iqr_proximity_rule', 
                   #'gaussian_approximation','quantiles'
                   fill_with_value = 'extreme_values',
                   #'zeros','mean-median'
                   targetencoding = False,
                   save_to_csv = False,
                  ):
        X = dd.from_pandas(X, npartitions=self.n_gpus)
        self.initialize_types(X)      
        workflow = nvt.Workflow(cat_names=self.categor_columns, 
                        cont_names=self.numer_columns,
                        label_name=self.y_names,
                        client=self.client                       )
        # Operators: https://nvidia.github.io/NVTabular/main/api/ops/index.html      
        # Categorify https://nvidia.github.io/NVTabular/main/api/ops/categorify.html
        if encode_categor_type == 'categorify':
            if len(self.categor_columns) != 0:
                workflow.add_preprocess(
                    ops.Categorify(columns = self.categor_columns, out_path='./data/')                )
        
        if encode_categor_type == 'onehotencoding':
            #TODO: FIX the BUG!
            X_cat_encoded = OneHotEncoder().fit_transform(X[self.categor_columns].to_dask_array(lengths=True))
#            display(X_cat_encoded)

            #lengths=True - chunk sizes can be computed
            X = X.drop(self.categor_columns, axis=1)
            X = X.append(dd.io.from_dask_array(X_cat_encoded))          
            display(X)
            print(type(X))
            self.initialize_types(X)            

        # OutlDetect https://nvidia.github.io/NVTabular/main/api/ops/clip.html
        if len(self.numer_columns) != 0:
            lower, upper = self.outldetect(outliers_detection_technique, X[self.numer_columns])
            for i in range(len(self.numer_columns)):
                workflow.add_preprocess(
                    ops.Clip(min_value=lower[i], max_value=upper[i], columns=[self.numer_columns[i]])                )
        # FillMissing https://nvidia.github.io/NVTabular/main/api/ops/fillmissing.html
        if fill_with_value == 'zeros':
            workflow.add_preprocess(
                ops.FillMissing(fill_val=0, columns=self.categor_columns + self.numer_columns)            )        
        if fill_with_value == 'extreme_values':
            extrim_values = {}
            if len(self.numer_columns) != 0:
                extrim_values.update(self.extrvalsdetect(X[self.numer_columns], 'numer_columns'))
            if len(self.categor_columns) != 0:
                extrim_values.update(self.extrvalsdetect(X[self.categor_columns], 'categor_columns'))    
            print('\n extrim_values:', extrim_values)
            output = open('./data/' + 'extrim_values', 'wb')
            pickle.dump(extrim_values, output)
            output.close()
            for fill_val, column in zip(list(extrim_values.values()), list(extrim_values.keys())):
                workflow.add_preprocess(
                    ops.FillMissing(fill_val=fill_val, columns=[column])                )
        if fill_with_value == 'mean-median':
            if len(self.categor_columns) != 0:              
                workflow.add_preprocess(
                        ops.FillMedian(columns=self.categor_columns, preprocessing=True, replace=True)                    ) 
            if len(self.numer_columns) != 0:
                means = list(dd.from_pandas(X[self.numer_columns], 
                                            npartitions=self.n_gpus).mean().compute().values)
                for fill_val, column in zip(means, self.numer_columns):
                    workflow.add_preprocess(
                        ops.FillMissing(fill_val=fill_val, columns=[column])                    )
                    
        if targetencoding:
            print('\n TODO: targetencoding')
            #https://nvidia.github.io/NVTabular/main/api/ops/targetencoding.html

        
        #######################################################        
        workflow.finalize()
        dataset = nvt.Dataset(X)
        tmp_output_path="./parquet_data_tmp"
        workflow.apply(
             dataset,
             output_format="parquet",
             output_path=tmp_output_path,
             shuffle=Shuffle.PER_WORKER,  # Shuffle algorithm
             out_files_per_proc=8, # Number of output files per worker
             )
        files = glob.glob(tmp_output_path + "/*.parquet")
        X_final = cudf.read_parquet(files[0])
        for i in range(1, len(files)):    
            X_final = X_final.append(cudf.read_parquet(files[i]))      
        # Delete temporary files
        shutil.rmtree(tmp_output_path, ignore_errors=True)       
        if save_to_csv:
            try:
                X_final.to_csv('./data/' + self.filename, index=False)
            except TypeError:
                print('Initialize filename!')
        return X_final 
            