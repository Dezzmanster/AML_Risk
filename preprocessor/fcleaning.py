# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:06:05 2020

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

class EmptyElim(object):
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
    
    path: str, default = None. File path to save dataframe. 
        None - does not save

    '''
    def __init__(self, n_jobs = None, 
                 chunks = None, path = None):
        if n_jobs == None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs        
        self.chunks = chunks
        self.path = path

    def detect_col(self, X):        
        for column in X.columns:
            if X[column].nunique() < 2:
                self.col_names[column] = list(X[column].unique()) 
        return self.col_names
    
    def drop_col(self, X):
        columns = [i for i in list(self.col_names.keys()) if i in list(X.columns)]
        X.drop(columns=columns, inplace=True)
        return X

    def fit(self, X):        
        self.col_names = {}   
        if self.chunks == None:
           self.chunks  = int(len(X.columns)/self.n_jobs)
        return_ = Pool(processes = self.n_jobs).map(self.detect_col, 
                             [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                           )     
        for r in return_:
          self.col_names.update(r)
        print('\n col_names:', self.col_names) 

    def transform(self, X):     
        if self.chunks == None:
           self.chunks  = int(len(X.columns)/self.n_jobs)
        X = pd.concat(Pool(processes = self.n_jobs).map(self.drop_col, 
                             [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                           ), axis=1)
        if self.path != None:
            X.to_csv(self.path)
        return X    
        
    def fit_transform(self, X):
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
    
    path: str, default = None. File path to save dataframe. 
        None - does not save
    '''

    def __init__(self, outliers_detection_technique = 'iqr_proximity_rule', n_jobs = None, 
                 chunks = None, path = None):            
        self.outliers_detection_technique = outliers_detection_technique
        if n_jobs == None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs        
        self.chunks = chunks
        self.path = path
                  
    def iqr_proximity_rule(self, X):
        for column in X.columns:
            x = X[column]
            IQR = x.quantile(0.75) - x.quantile(0.25)
            lower = x.quantile(0.25) - (IQR * 1.5)
            upper = x.quantile(0.75) + (IQR * 1.5)
            self.col_outl_info[column] = (lower, upper)
        return self.col_outl_info
    
    def iqr_proximity_rule_replace(self, X):
        for column in X.columns:
          x = X[column]
          self.col_outl_info[column]
          lower, upper = self.col_outl_info[column]
          X[column] = np.where(x > upper, upper, np.where(x < lower, lower, x))
        return X
    
    def gaussian_approximation(self, x):
        # Gaussian approximation
        #TODO
        return x
    
    def quantiles(self, x):
        # Using quantiles
        #TODO
        return x

    def fit(self, X):
        self.col_outl_info = {}

        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)

        if self.outliers_detection_technique == 'iqr_proximity_rule':
            return_ = Pool(processes = self.n_jobs).map(self.iqr_proximity_rule, 
                                [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                )
        for r in return_:
            self.col_outl_info.update(r)          
        print('\n col_outl_info:', self.col_outl_info) 

    
    def transform(self, X):
         if self.chunks == None:
           self.chunks  = int(len(X.columns)/self.n_jobs)
         if self.outliers_detection_technique == 'iqr_proximity_rule':
            X =  pd.concat(Pool(processes = self.n_jobs).map(self.iqr_proximity_rule_replace, 
                             [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                             ), axis=1)
         if self.path != None:
            X.to_csv(self.path)
         return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)