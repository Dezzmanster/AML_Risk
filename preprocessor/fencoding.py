# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:11:40 2020

@author: Anna
"""

import os, copy, time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

class Dtimetodata(object):
    '''
    dtime_col_names: dict, default = None. Dictionary of columns that should be transformed into time columns.
        None - will not be executed 
        example: {
                  'ddays' : list of columns,
                  'dmonths' : list of columns,
                  'dyears' : list of columns
                  }
    
    start_date: str, default = '2020-01-01'. The starting point.

    time_encode: bool, default = False. If True, then time columns will be encoded via sklearn functions: .dt.year .dt.month, .dt.day, and added to the DataFrame.

    drop_current: bool, default = False. If True, then all time columns will be dropped. 

    n_jobs: int, default = None. The number of jobs to run in parallel.
        None - means 1
        -1 - means using all processors
    
    chunks: int, default = None. Number of features sent to processor per time.
        None - means number of features/number of cpu
    
    path: str, default = None. File path to save dataframe. 
        None - does not save
    
    '''
    
    def __init__(self, dtime_col_names = None, start_date = '2020-01-01', 
                 time_encode = False, drop_current = False, 
                 n_jobs = None, chunks = None, path = None):
      
        self.dtime_col_names = dtime_col_names
        self.start_date = pd.Timestamp(start_date)
        self.time_encode = time_encode
        self.drop_current = drop_current
      
        if n_jobs == None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs        
        self.chunks = chunks
        self.path = path

    def dtime_to_data(self, X):
        if self.dtime_col_names != None:
            for delays, k in zip(list(self.dtime_col_names.keys()), [1, 30, 365]):
                for column in self.dtime_col_names[delays]:
                    if any(column == c for c in X.columns):
                        #print('\n {} processed ...'.format(column))
                        X[column + '_date'] = self.start_date + pd.to_timedelta(X[column]*k, 'D')
                        if self.drop_current:
                            X.drop(columns=[column], inplace=True)

        if self.time_encode:        
            for column in X.columns:
              if str(X[column].dtype) == 'datetime64[ns]': 
                  # TODO: check if there are any other time types
                  # datetime64[ns] -> int64
                  X[column + '_year'] = X[column].dt.year
                  X[column + '_month'] = X[column].dt.month
                  X[column + '_day'] = X[column].dt.day
                  #print('\n {} was encoded'.format(column))
                  if self.drop_current:                      
                      X.drop(columns=[column], inplace=True)                  
        return X 

    def transform(self, X):   
        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)
        X =  pd.concat(Pool(processes = self.n_jobs).map(self.dtime_to_data, 
                                [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                ), axis=1)
        if self.path != None:
              X.to_csv(self.path)
        return X
    
class FEncoding(object):
    def __init__(self, n_jobs = 1, chunks = None, path = None):      
        
        self.categor_types = ['object', 'bool', 'int32', 'int64']
        self.numer_types = ['float', 'float32', 'float64']
        self.time_types = ['datetime64[ns]'] # What else?

        if n_jobs == None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs        
        self.chunks = chunks
        self.path = path  

    def initialize_types_(self, X):
        # Sometimes categorical feature can be presented with a float type.
        # Sometimes numerical feature can be presented with an int type.  
        # Let's check for that  
        for column in X.columns:
            c_type = str(X[column].dtype) 
            unique_values_X = X[column].unique()        
            if any(c_type == t for t in self.numer_types) & (len(unique_values_X) < 20):
                print('\n {} has type {} and number of unique values: {}, will be considered as a categorical \n'.format(column, c_type, len(unique_values_X)))
                self.categor_columns.append(column)            
            elif any(c_type == t for t in self.categor_types) & (len(unique_values_X) > 20):
                print('\n {} has type {} and number of unique values: {}, will be considered as a numerical \n'.format(column, c_type, len(unique_values_X)))
                self.numer_columns.append(column)
            elif any(c_type == t for t in self.categor_types):
                self.categor_columns.append(column)
            elif any(c_type == t for t in self.time_types):
                self.time_columns.append(column)            
            else:
                self.numer_columns.append(column)
        return self.categor_columns, self.numer_columns, self.time_columns
    
    def bucket_numerical_(self, X):
        # TODO: specify or introduce a criterion which columns to bake
        # K-bins discretization based on quantization

        def get_input_keypoints(f_data, n_kps):
            while len(np.quantile(f_data, np.linspace(0.0, 1.0, num=n_kps))) != len(np.unique(np.quantile(f_data, np.linspace(0.0, 1.0, num=n_kps)))):
                n_kps -= 1
            return np.quantile(f_data, np.linspace(0.0, 1.0, num=n_kps))

        if self.columns_to_buck == 'all_numerical':
            self.columns_to_buck = self.numer_columns            
        if type(self.columns_to_buck) != list:
            raise VlaueError('Identify list of columns_to_buck')

        for column in X.columns:
            if any(column == col for col in self.columns_to_buck):
                print('\n {} bucketing ...'.format(column))
                f_X = X[column].values.ravel()
                X[column + '_bucketed'] = np.digitize(f_X, get_input_keypoints(f_X, self.n_bins))
                if self.drop_current:
                    X.drop(columns=[column], inplace=True)
        return X

    def encode_categor_(self, X, method = 'OrdinalEncoder'):
        

        if self.method == 'OrdinalEncoder':
            enc = preprocessing.OrdinalEncoder()
            X = pd.DataFrame(enc.fit_transform(X), columns = X.columns)

        if self.method == 'OneHotEncoder':            
            X = pd.get_dummies(X, drop_first=True, dummy_na=True)
        
        return X

    
    def initialize_types(self, X):
        self.categor_columns, self.numer_columns, self.time_columns = [], [], []
        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)
        return_ = Pool(processes = self.n_jobs).map(self.initialize_types_, 
                                 [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                 )
        
        for i in range(len(return_)):
            categor_columns, numer_columns, time_columns = return_[i]
            self.categor_columns += categor_columns
            self.numer_columns += numer_columns
            self.time_columns += time_columns
        return {'categor_columns': self.categor_columns,
                           'numer_columns': self.numer_columns,
                           'time_columns': self.time_columns       
         }

    def bucket_numerical(self, X, 
                         n_bins=5, columns_to_buck = 'all_numerical', 
                         drop_current = False):      
        self.n_bins = n_bins
        self.columns_to_buck = columns_to_buck
        self.drop_current = drop_current
        self.initialize_types(X)
        if self.chunks == None:
              self.chunks  = int(len(X.columns)/self.n_jobs)
        X =  pd.concat(Pool(processes = self.n_jobs).map(self.bucket_numerical_, 
                                [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                ), axis=1)
        if self.path != None:
              X.to_csv(self.path)
        return X

    def encode_categor(self, X, method = 'OrdinalEncoder'):
        self.method = method
        self.initialize_types(X)        
        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)

        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        
        X_cat =  pd.concat(Pool(processes = self.n_jobs).map(self.encode_categor_, 
                                [X[categor_columns[start: start + self.chunks]] for start in range(0, len(categor_columns), self.chunks)]
                                ), axis=1)
        X = pd.concat([X[numer_columns],X_cat], axis=1)    
        
        if self.path != None:
              X.to_csv(self.path)
        return X


