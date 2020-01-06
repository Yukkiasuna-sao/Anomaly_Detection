import pandas
import numpy as np

from sklearn import metrics
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping

import gc
import warnings
warnings.filterwarnings('ignore')

class UncompleteAutoencoder:
    def __init__(self, df):
        self.df = df
        self.input_dim = self.df.shape[1]
    
    def Modeling(self, train, dense_dim, batchsize = None, validation_size = None):
        if batchsize == None:
            raise AssertionError("Batchsize must be defined.")
        self.train = train
        self.dense_dim = dense_dim
        
        model = Sequential()
        model.add(Dense(self.dense_dim, input_dim = self.input_dim, activation = 'relu'))
        model.add(Dense(self.input_dim))
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        print(model.summary())
        
        self.model = model
        
        self.model.fit(train, train, batch_size = batchsize, validation_split = validation_size,
                       verbose = 1, epochs = 50, callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3)])
        
        gc.collect()
        
    def Prediction(self, test_data, data_type = None):
        self.test_data = test_data
        if data_type == None:
            raise AssertionError('Data type must be defined.')
            
        elif data_type == 'Insample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Insample Normal Score (RMSE) : {}".format(score))
        
        elif data_type == 'OutOfSample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print('Out of Sample Normal Score (RMSE) : {}'.format(score))
        
        elif data_type == 'Attack':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print('Attack Underway Score (RMSE) : {}'.format(score))

"""
tmp = UncompleteAutoencoder(x_normal_train)
tmp.Modeling(x_normal_train, 25, batchsize = 50, validation_size = 0.25)

tmp.Prediction(x_normal_test, data_type = 'Insample')
"""


