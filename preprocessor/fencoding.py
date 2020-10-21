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


