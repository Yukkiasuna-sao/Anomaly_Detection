"""
Outlier Detection Case

The training data contains outliers which are defined as observations that are far from the others. Outlier detection estimators thus try to fit the regions where the training data is the most concentrated, ignoring the deviant observations.

So, The dataset must contain normal and Anomalies.

TO - DO


"""

import numpy as np

from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings('ignore')
import gc

class SimpleDBSCAN:
    def __init__(self, df):
        self.df = df
        
    def Modeling(self, train_data, epsilon, min_samples, seed):
        self.train_data = train_data
        self.seed = seed
        self.epsilon = epsilon
        self.min_samples = min_samples
        
        model = DBSCAN(eps = self.epsilon, min_samples = self.min_samples).fit(self.train_data)
        
        self.model = model
        
        gc.collect()
    def Prediction(self, test_data, data_type):
        self.test_data = test_data
        
        def ConvertLabel(x):
            if x == -1:
                return 1
            
            else:
                return 0
        
        function = np.vectorize(ConvertLabel)
        
        if data_type == None:
            raise AssertionError('Data Type must be defined')
            
        elif data_type == 'Insample':
            pred = self.model.predict(self.test_data)
            pred = function(pred)
            pred = list(pred)
            
            print('Insample Classification Result \n')
            print('Normal Values : {}'.format(pred.count(0)))
            print('Anomalies Values {}'.format(pred.count(1)))
            
        elif data_type == 'OutOfSample':
            pred = self.model.predict(self.test_data)
            pred = function(pred)
            pred = list(pred)
            
            print('Insample Classification Result \n')
            print('Normal Values : {}'.format(pred.count(0)))
            print('Anomalies Values {}'.format(pred.count(1)))
            
            return pred