# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:26:03 2020

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


class FCleaning(object):
    '''
    Outliers detection techniques:
        iqr_proximity_rule - the boundaries are determined using IQR proximity rules
        gaussian_approximation - sets the boundaries with values according to the mean and standard deviation
        quantiles - the boundaries are determined using the quantiles, through which you can specify any percentage you want

    Source: https://heartbeat.fritz.ai/hands-on-with-feature-engineering-techniques-dealing-with-outliers-fcc9f57cb63b

    '''

    def __init__(self, outliers_detection_technique = None, 
                 n_jobs = 1, chunks = None, 
                 path = None,
                 ):
      
        self.outliers_detection_technique = outliers_detection_technique
        self.n_jobs = n_jobs
        self.chunks = chunks
        self.path = path


    
    def emptyness_elimination_(self, X):
        for column in X.columns:
            if X[column].nunique() < 2:
                print('\n {}, unique values: {}'.format(column, X[column].unique()))
                X = X.drop(columns=[column])         
        return X
        
    def emptyness_elimination(self, X):
        if self.chunks == None:
           self.chunks  = int(len(X.columns)/self.n_jobs)
        p = Pool(processes = self.n_jobs)
        X = pd.concat(p.map(self.emptyness_elimination_, 
                             [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                           ), axis=1)
        if self.path != None:
            X.to_csv(self.path)
        return X
                        
    
    def iqr_proximity_rule(self, x):
        # Inter-quantal range proximity rule
        # Calculate the IQR
        if str(x.dtype) != 'object': #FIX IT
            IQR = x.quantile(0.75) - x.quantile(0.25)

            # Calculate the boundries
            lower = x.quantile(0.25) - (IQR * 1.5)
            upper = x.quantile(0.75) + (IQR * 1.5)

            # Replacing the outliers
            x = np.where(x > upper, upper, np.where(x < lower, lower, x))

        return x
    
    def gaussian_approximation(self, x):
        # Gaussian approximation
        #TODO
        return x
    
    def quantiles(self, x):
        # Using quantiles
        #TODO
        return x
    
    def outliers_elimination_(self, X):
        if self.outliers_detection_technique == 'iqr_proximity_rule':
            for column in X.columns:
                X[column] = self.iqr_proximity_rule(X[column])
        return X
    
    def outliers_elimination(self, X):
         if self.chunks == None:
           self.chunks  = int(len(X.columns)/self.n_jobs)
         p = Pool(processes = self.n_jobs)
         X =  pd.concat(p.map(self.outliers_elimination_, 
                             [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                             ), axis=1)
         if self.path != None:
            X.to_csv(self.path)
         return X

class FEncoding(object):
    def __init__(self, 
                 n_jobs = 1, chunks = None, 
                 path = None,
                 ):      
        self.n_jobs = n_jobs
        self.chunks = chunks
        self.path = path
        self.categor_types = ['object', 'bool', 'int32', 'int64']
        self.numer_types = ['float', 'float32', 'float64']
        self.time_types = ['datetime64[ns]']
    
    def dtime_to_data_(self, X):
        if self.dtime_col_names != None:
            for delays, k in zip(list(self.dtime_col_names.keys()), [1, 30, 365]):
                for column in self.dtime_col_names[delays]:
                    if any(column == c for c in X.columns):
                        print('\n {} processed ...'.format(column))
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
                  print('\n {} was encoded'.format(column))
                  if self.drop_current:                      
                      X.drop(columns=[column], inplace=True)                  
        return X    
    
    def pick_categor_(self, X):
        categor_columns = []
        numer_columns = []
        time_columns = []
        # Sometimes categorical feature can be presented with a float type. 
        # Let's check for that
        n_unique = 20          
        for column in X.columns:
            c_type = str(X[column].dtype) 
            unique_values_X = X[column].unique()        
            if any(c_type == t for t in self.numer_types) & (len(unique_values_X) < n_unique):
                print('\n {} has type {} and unique values: {} -> {}, will be considered as categorical \n'.format(column, c_type, unique_values_X))
                categor_columns.append(column)
            elif any(c_type == t for t in self.categor_types):
                categor_columns.append(column)
            elif any(c_type == t for t in self.time_types):
                time_columns.append(column)            
            else:
                numer_columns.append(column)

        return categor_columns, numer_columns, time_columns
    
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
        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        if self.method == 'OrdinalEncoder':
            enc = preprocessing.OrdinalEncoder()
            X[categor_columns] = enc.fit_transform(X[categor_columns])

        if self.method == 'OneHotEncoder':
            X.append(pd.get_dummies(X[self.categor_columns], drop_first=True, dummy_na=True))
            X.drop(columns=[categor_columns], inplace=True)

        return X
    
    def dtime_to_data(self, X, 
                        dtime_col_names = None,           
                        start_date = '2020-01-01', 
                        time_encode = False, 
                        drop_current = False):
      
        self.dtime_col_names = dtime_col_names
        self.start_date = pd.Timestamp(start_date)
        self.time_encode = time_encode
        self.drop_current = drop_current

        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)
        p = Pool(processes = self.n_jobs)
        X =  pd.concat(p.map(self.dtime_to_data_, 
                                [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                ), axis=1)
        if self.path != None:
              X.to_csv(self.path)
        return X

    def pick_categor(self, X):
        self.categor_columns = []
        self.numer_columns = []
        self.time_columns = []
        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)
        p = Pool(processes = self.n_jobs)

        return_ = p.map(self.pick_categor_, 
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
                         n_bins=5, 
                         columns_to_buck = 'all_numerical', 
                         drop_current = False):
      
        self.n_bins = n_bins
        self.columns_to_buck = columns_to_buck
        self.drop_current = drop_current

        self.pick_categor(X)

        if self.chunks == None:
              self.chunks  = int(len(X.columns)/self.n_jobs)
        p = Pool(processes = self.n_jobs)   
        X =  pd.concat(p.map(self.bucket_numerical_, 
                                [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                ), axis=1)
        if self.path != None:
              X.to_csv(self.path)
        return X

    def encode_categor(self, X, method = 'OrdinalEncoder'):
        self.method = method

        self.pick_categor(X)
        
        if self.chunks == None:
            self.chunks  = int(len(X.columns)/self.n_jobs)

        p = Pool(processes = self.n_jobs)
        X =  pd.concat(p.map(self.encode_categor_, 
                                [X[list(X.columns)[start: start + self.chunks]] for start in range(0, len(X.columns), self.chunks)]
                                ), axis=1)
        if self.path != None:
              X.to_csv(self.path)
        return X
