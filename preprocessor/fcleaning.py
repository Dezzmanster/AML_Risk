# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:06:05 2020
@author: Anna
"""
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
logging.config.fileConfig(fname='logger.ini', defaults={'logfilename': 'logfile.log'})
import warnings
warnings.filterwarnings('ignore')

def save_to_csv(X, rest_columns=None, path=None):
    '''
    X: pd.DataFrame. The main table.
    rest_columns: pd.Series. The rest columns to concat with X before saving.
    path: string. The path where to save the table.
    '''
    if path == None:
        path = 'trial_{}.csv'.format(time.strftime("%m%d%Y-%H:%M"))

    if rest_columns != None:
        pd.concat([X, rest_columns], axis=1).to_csv(path, index=False)
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
    start_mem = X.memory_usage().sum() / 1024**2
    print('\n Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    logging.info(f"Memory usage of dataframe is {start_mem} MB'")
    for col in X.columns:
            col_type = X[col].dtype
            if col_type != object:
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
            else:
                X[col] = X[col].astype('category')
    end_mem = X.memory_usage().sum() / 1024**2
    print('\n Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('\n Memory usage decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
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
    @timeit
    def __init__(self, n_jobs = None, chunks = None):
        if n_jobs == None:
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

    @timeit
    def fit(self, X):
        '''
        Create a dictionary of names of columns (self.colnames) to drop.
        '''        
        self.col_names = {} 
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)
        return_ = mp.Pool(processes = self.n_jobs).map(self.detect_col_, 
                             [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                           )     
        for r in return_:
          self.col_names.update(r)
        print('\n col_names:', C)
        logging.info(f"Dictionary of names of columns to drop has been created: {self.colnames}")
        
    @timeit
    def transform(self, X):
        '''
        Drops self.col_names that were initialized by self.fit function.
        '''   
        logging.info(f"Initial X shape: {X.shape}")
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)
        X = pd.concat(mp.Pool(processes = self.n_jobs).map(self.drop_col_, 
                             [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                           ), axis=1)
        logging.info(f"Columns were dropped, X new shape: {X.shape}")
        return X     
    
    @timeit    
    def fit_transform(self, X):
        '''
        Find self.col_names and drops them.
        '''
        self.fit(X)
        return self.transform(X)



class OutlDetect(object):
    '''
    outliers_detection_technique: {'iqr_proximity_rule', 'gaussian_approximation', 'quantiles'}, default = 'iqr_proximity_rule'
        'iqr_proximity_rule' - the boundaries are determined using IQR proximity rules
        'gaussian_approximation' - sets the boundaries with values according to the mean and standard deviation
        'quantiles' - the boundaries are determined using the quantiles, through which you can specify any percentage you want
         Source: https://heartbeat.fritz.ai/hands-on-with-feature-engineering-techniques-dealing-with-outliers-fcc9f57cb63b
    
    n_jobs: int, default = None. The number of jobs to run in parallel.
        None - means 1
        -1 - means using all processors
    
    chunks: int, default = None. Number of features sent to processor per time.
        None - means number of features/number of cpu
    '''
    
    @timeit
    def __init__(self, outliers_detection_technique = 'iqr_proximity_rule', n_jobs = None, 
                 chunks = None):            
        self.outliers_detection_technique = outliers_detection_technique
        if n_jobs == None:
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
    
    @timeit
    def fit(self, X):
        '''
        Collect information regarding self.col_outl_info - lower and upper bounds to clip outliers.
        '''
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)

        if self.outliers_detection_technique == 'iqr_proximity_rule':
            f = self.iqr_proximity_rule
        elif self.outliers_detection_technique == 'gaussian_approximation':
            f = self.gaussian_approximation
        elif self.outliers_detection_technique == 'quantiles':
            f = self.quantiles
        
        self.col_outl_info = {}
        return_ = mp.Pool(processes = self.n_jobs).map(f, 
                            [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                            )
        for r in return_:
            self.col_outl_info.update(r)          
        print('\n col_outl_info (upper, lower) bounds:', self.col_outl_info) 
        
        logging.info(f"{self.outliers_detection_technique}, col_outl_info (upper, lower) bounds:{self.col_outl_info}")
    
    @timeit
    def transform(self, X):   
        '''
        Clip ouliers by using the dict of lower and upper bounds (self.col_outl_info).
        '''
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)
          
        X =  pd.concat(mp.Pool(processes = self.n_jobs).map(self.replace, 
                              [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                              ), axis=1)
        logging.info(f"Successfully clipped")
        return X  
    
    @timeit
    def fit_transform(self, X):
        '''
        1. Collect information regarding self.col_outl_info - lower and upper bounds to clip outliers.
        2. Clip ouliers by using the dict of lower and upper bounds (self.col_outl_info).
        '''
        self.fit(X)
        return self.transform(X)