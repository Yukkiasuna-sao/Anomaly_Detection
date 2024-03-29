"""
Novelty Detection Case

The training data is not polluted by outliers and we are interested in detecting whether a new observation is an outlier. In this context an outlier is also called a novelty.

"""
import numpy as np

from sklearn.svm import  OneClassSVM

class SimpleOneClassSVM:
    def __init__(self, df):
        self.df = df
        
    def Modeling(self, train_data, seed):
        self.train_data = train_data
        self.seed = seed
         
        model = OneClassSVM(nu = 0.01, random_state = self.seed).fit(self.train_data) # TO - DO: Hyperparameter Tuning
        
        self.model = model
    
    def Prediction(self, test_data, data_type):
        self.test_data = test_data
        
        def ConvertLabel(x):
            if x == -1:
                return 1
    
            else:
                return 0
            
        function = np.vectorize(ConvertLabel)
            
        if data_type == None:
            raise AssertionError('Data Type must be defined.')
            
        elif data_type == 'Insample':
            pred = self.model.predict(self.test_data)
            pred = function(pred)
            pred = list(pred)
            
            print('Insample Classification Result \n')
            print('Normal Value: {}'.format(pred.count(0)))
            print('Anomaly Value: {}'.format(pred.count(1)))

        elif data_type == 'OutOfSample':
            pred = self.model.predict(self.test_data)
            pred = function(pred)
            pred = list(pred)
            
            print('OutOfSample Classification Result \n')
            print('Normal Value: {}'.format(pred.count(0)))
            print('Anomaly Value: {}'.format(pred.count(1)))
            
        elif data_type == 'Attack':
            pred = self.model.predict(self.test_data)
            pred = function(pred)
            pred = list(pred)
            
            print('Anomaly Classification Result \n')
            print('Normal Value: {}'.format(pred.count(0)))
            print('Anomaly Value: {}'.format(pred.count(1)))
            
            self.pred = pred
            
            return self.pred

