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
from sklearn.preprocessing import KBinsDiscretizer
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
