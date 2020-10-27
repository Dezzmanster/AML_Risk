# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:11:40 2020
@author: Anna
"""
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
 

import logging
# logger = logging.getLogger()
# fhandler = logging.FileHandler(filename='fencoding_log.log', mode='a')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fhandler.setFormatter(formatter)
# logger.addHandler(fhandler)
# logger.setLevel(logging.DEBUG)
    
class FEncoding(object):
    
    def __init__(self, n_jobs = 1, chunks = None):      
        
        self.categor_types = ['object', 'bool', 'int32', 'int64', 'int8']
        self.numer_types = ['float', 'float32', 'float64']
        self.time_types = ['datetime64[ns]', 'datetime64[ns, tz]'] 
        # What else? https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        # TODO: check if there are any other time types

        if n_jobs == None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs        
        self.chunks = chunks
        
        logging.info(f"Object {self} is created")
    
    def date_replace_(self, X):
        def pars_date(x):
            fmts = ('%Y', '%b %d, %Y','%b %d, %Y','%B %d, %Y','%B %d %Y','%m/%d/%Y','%m/%d/%y','%b %Y','%B%Y','%b %d,%Y', 
                      '%d.%m.%Y', '%Y.%m.%d', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S')
            t = True
            if str(x.dtype) == 'object':
              for fmt in fmts:
                  try:
                      return pd.Series([dt.datetime.strptime(str(x.iloc[i]), fmt) for i in range(len(x))]).apply(lambda q: q.strftime('%m/%d/%Y')).astype('datetime64[ns]')
                      t = False
                      break 
                  except ValueError as err:
                      pass
            if t and (len(str(x.iloc[0])) > 9) and (len(str(x.iloc[0])) <= 14): 
            # TODO: better condition on string to identify that it is unix timestep
              try:
                  x = x.astype('float')
                  return pd.Series([dt.datetime.fromtimestamp(x.iloc[i]) for i in range(len(x))]).apply(lambda q: q.strftime('%m/%d/%Y')).astype('datetime64[ns]')
              except ValueError as err:
                  pass
        
        for column in X.columns:
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
        for column in X.columns:
            c_type = str(X[column].dtype) 
            unique_values_X = X[column].unique() 
            if any(c_type == t for t in self.numer_types):
                unique_values = np.unique(X[column][~np.isnan(X[column])])
                if np.array([el.is_integer() for el in unique_values]).sum() == len(unique_values):
                    print('\n {} has type {} and number of unique values: {}, will be considered as a categorical \n'.format(column, c_type, len(unique_values_X)))
                    self.categor_columns.append(column)
                else:
                    self.numer_columns.append(column)
            if any(c_type == t for t in self.categor_types):
                self.categor_columns.append(column)
            if any(c_type == t for t in self.time_types):
                self.time_columns.append(column)                          

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
                keypoints = get_input_keypoints(f_X, self.n_bins)
                if len(pd.Series(keypoints).value_counts()) <= 1:
                    print('\n', column, 'has keypoints:', keypoints, ', and can not be bucketed.')
                    pass
                else:
                    X[str(column) + '_bucketed'] = np.digitize(f_X, keypoints)
                    if self.drop_current:
                        X.drop(columns=[column], inplace=True)
            else:
                print('\n ', column, 'is not numerical!')
              
        return X

    def encode_categor_(self, X, method = 'OrdinalEncoder'):    
        if self.method == 'OrdinalEncoder':
            mask = X.isnull().to_numpy()
            X_original = X
            X = X.astype(str)
            

            enc = preprocessing.OrdinalEncoder(dtype='int')
            X = pd.DataFrame(enc.fit_transform(X), columns = X.columns)
            X = X.mask(mask, X_original)

        if self.method == 'OneHotEncoder':  
            X = X.astype(object)
            X = pd.get_dummies(X)

        return X

    def encode_time_(self, X):
        for column in X.columns:
            if any(str(X[column].dtype) == t for t in self.time_types):
                X[str(column) + '_year'] = X[column].dt.year
                X[str(column) + '_month'] = X[column].dt.month
                X[str(column) + '_day'] = X[column].dt.day
                print('\n {} was encoded from date'.format(column))
                if self.drop_current:                      
                    X.drop(columns=[column], inplace=True)
        return X  
    
    
    def initialize_types(self, X):    
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)
        
        self.categor_columns, self.numer_columns, self.time_columns = [], [], []
        return_ = mp.Pool(processes = self.n_jobs).map(self.initialize_types_, 
                                 [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                                 )
        
        for i in range(len(return_)):
            categor_columns, numer_columns, time_columns = return_[i]
            self.categor_columns += categor_columns
            self.numer_columns += numer_columns
            self.time_columns += time_columns
        
        logging.info(f"Types have been initialized.")
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

        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]

        n_columns_X = len(numer_columns)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)

        X_num =  pd.concat(mp.Pool(processes = self.n_jobs).map(self.bucket_numerical_, 
                                [X[numer_columns[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                                ), axis=1)

        rest_col = categor_columns + time_columns
        if len(rest_col) != 0:
            X = pd.concat([X[rest_col], X_num], axis=1)
        else:
            X = X_num
        
        logging.info(f"Columns {numer_columns} have been bucketed.")
        return X

    
    def encode_categor(self, X, method = 'OrdinalEncoder'):
        self.method = method
        self.initialize_types(X)

        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]

        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(len(categor_columns)/self.n_jobs)

        X_cat =  pd.concat(mp.Pool(processes = self.n_jobs).map(self.encode_categor_, 
                                [X[categor_columns[start: start + self.chunks]] for start in range(0, len(categor_columns), self.chunks)]
                                ), axis=1)

        rest_col = numer_columns + time_columns
        if len(rest_col) != 0:
            X = pd.concat([X[rest_col], X_cat], axis=1)
        else:
            X = X_cat  
        
        logging.info(f"Columns {categor_columns} have been encoded.")
        return X

    
    def encode_time(self, X, drop_current = False):
        self.drop_current = drop_current
        self.initialize_types(X)

        categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
        numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
        time_columns = [i for i in self.time_columns if i in list(X.columns)]

        n_columns_X = len(time_columns)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
               self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)

        X_time =  pd.concat(mp.Pool(processes = self.n_jobs).map(self.encode_time_, 
                                [X[time_columns[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                                ), axis=1)
        
        rest_col = numer_columns + categor_columns
        if len(rest_col) != 0:
            X = pd.concat([X[rest_col], X_time], axis=1)
        else:
            X = X_time    
        logging.info(f"Columns {time_columns} have been encoded.")
        return X

    
    def date_replace(self, X):
        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)
   
        X =  pd.concat(mp.Pool(processes = self.n_jobs).map(self.date_replace_, 
                                [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                                ), axis=1)  
    
        print('\n time_columns:', self.initialize_types(X)['time_columns']) 
        logging.info(f"Columns {self.initialize_types(X)['time_columns']} have been detected as date and encoded as yy-mm-dd.")
        return X 

class FImputation(FEncoding):
    '''
      Dealing with missing values:
      We will use simple techniques with regards to the model that we use.
      For tree-based models, nana will be filled in with max values (or zeros)
      For regression with means and medians for numerical and categorical types respectively.
    '''
    
    def __init__(self, model_type, fill_with_value = None, 
                  n_jobs = None, chunks = None):
        super(FImputation, self).__init__()
        
        self.model_type = model_type
        self.fill_with_value = fill_with_value
        
        if n_jobs == None:
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
                raise VlaueError('Identify fill_with_value parameter')

        if self.model_type == 'regression-based':
            categor_columns = [i for i in self.categor_columns if i in list(X.columns)]
            numer_columns = [i for i in self.numer_columns if i in list(X.columns)]
            for column in X.columns:
                unique_values = np.unique(X[column][~np.isnan(X[column])])
                if any(column == t for t in categor_columns):
                    X[column].fillna(int(np.median(unique_values)), inplace=True)
                if any(column == t for t in numer_columns):
                    X[column].fillna(np.mean(unique_values), inplace=True)
            return X 
        
    
    def impute(self, X):
        #self.initialize_types(X)
        X = self.encode_categor(X, method = 'OrdinalEncoder')

        columns_X = list(X.columns)
        n_columns_X = len(columns_X)
        if self.chunks == None:
            while int(n_columns_X/self.n_jobs) <= 1:
                self.n_jobs -=1
            self.chunks = int(n_columns_X/self.n_jobs)

        X =  pd.concat(mp.Pool(processes = self.n_jobs).map(self.impute_, 
                                [X[columns_X[start: start + self.chunks]] for start in range(0, n_columns_X, self.chunks)]
                                ), axis=1)
        logging.info(f"Successfully imputed.")
        return X 

