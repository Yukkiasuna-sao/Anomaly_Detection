import numpy as np

from sklearn import metrics

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from keras.callbacks import EarlyStopping

import gc
import warnings
warnings.filterwarnings('ignore')

class SimpleUncompleteAutoencoder:
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
            
            return pred
        
        elif data_type == 'OutOfSample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print('Out of Sample Normal Score (RMSE) : {}'.format(score))
            
            return pred
        
        elif data_type == 'Attack':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print('Attack Underway Score (RMSE) : {}'.format(score))
            
            return pred


class SimpleStackedAutoencoder:
    def __init__(self, df):
        self.df = df
        self.input_dim = self.df.shape[1]
        
    def Modeling(self, train, hidden_dim = None, coding_dim = None, batchsize = None, validation_size = None):
        if hidden_dim == None:
            raise AssertionError("Hidden Layer Dimension must be defined.")
        if coding_dim == None:
            raise AssertionError("Coding Layer Dimension must be defined.")
        if batchsize == None:
            raise AssertionError("Batchsize must be defined.")
        
        self.train = train
        self.hidden_dim = hidden_dim
        self.coding_dim = coding_dim
        
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim = self.input_dim, activation = 'relu'))
        model.add(Dense(self.coding_dim, activation = 'relu'))
        model.add(Dense(self.hidden_dim, activation = 'relu'))
        model.add(Dense(self.input_dim))
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        print(model.summary())
        
        self.model = model
        
        self.model.fit(train, train, batch_size = batchsize, validation_split = validation_size,
                       verbose = 1, epochs = 50, callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3)])
        
        gc.collect()
        
    def Prediction(self, test_data, data_type):
        self.test_data = test_data
        
        if data_type == None:
            raise AssertionError('Data Type must be defined.')
        
        elif data_type == 'Insample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Insample Normal Score (RMSE) : {}".format(score))
            
            return pred
        
        elif data_type == 'OutOfSample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Out of Sample Normal Score (RMSE) : {}".format(score))
            
            return pred
            
        elif data_type =='Attack':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Attack Underway Score (RMSE) : {}".format(score))
            
            return pred
            

class SimpleDenosingAutoencoder:
    def __init__(self, df):
        self.df = df
        self.input_dim = self.df.shape[1]
        
    def Modeling(self, train, hidden_dim = None, coding_dim = None, batchsize = None, validation_size = None, denosing_type = None, std = None):
        if hidden_dim == None:
            raise AssertionError("Hidden Layer Dimension must be defined.")
        if coding_dim == None:
            raise AssertionError("Coding Layer Dimension must be defined.")
        if batchsize == None:
            raise AssertionError("Batchsize must be defined.")
        
        if denosing_type == None:
            raise AssertionError("Denosing Type must be Defined. ('Gaussian' or 'Dropout')")
        
        if denosing_type != None:
            if denosing_type == "Dropout":
                
                self.train = train
                self.hidden_dim = hidden_dim
                self.coding_dim = coding_dim
                
                model = Sequential()
                model.add(Dense(self.hidden_dim, input_dim = self.input_dim, activation = 'relu'))
                model.add(Dropout(0.2))
                model.add(Dense(self.coding_dim, activation = 'relu'))
                model.add(Dense(self.hidden_dim, activation = 'relu'))
                model.add(Dense(self.input_dim))
                
                model.compile(loss = 'mean_squared_error', optimizer = 'adam')
                print(model.summary())
                
                self.model = model
                
                self.model.fit(train, train, batch_size = batchsize, validation_split = validation_size,
                               verbose = 1, epochs = 50, callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3)])
                
                gc.collect()
            
            elif denosing_type == 'Gaussian':
                
                if std == None:
                    raise AssertionError('Gaussian Noise std must be defined.')
                
                self.train = train
                self.hidden_dim = hidden_dim
                self.coding_dim = coding_dim
                
                model = Sequential()
                model.add(Dense(self.hidden_dim, input_dim = self.input_dim, activation = 'relu'))
                model.add(GaussianNoise(std))
                model.add(Dense(self.coding_dim, activation = 'relu'))
                model.add(Dense(self.hidden_dim, activation = 'relu'))
                model.add(Dense(self.input_dim))
                
                model.compile(loss = 'mean_squared_error', optimizer = 'adam')
                print(model.summary())
                
                self.model = model
                
                self.model.fit(train, train, batch_size = batchsize, validation_split = validation_size,
                                verbose = 1, epochs = 50, callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3)])
                
                gc.collect()
                
        
    def Prediction(self, test_data, data_type):
        self.test_data = test_data
        
        if data_type == None:
            raise AssertionError('Data Type must be defined.')
        
        elif data_type == 'Insample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Insample Normal Score (RMSE) : {}".format(score))
            
            return pred
            
        elif data_type == 'OutOfSample':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Out of Sample Normal Score (RMSE) : {}".format(score))
            
            return pred
        
        elif data_type =='Attack':
            pred = self.model.predict(self.test_data)
            score = np.sqrt(metrics.mean_squared_error(pred, self.test_data))
            print("Attack Underway Score (RMSE) : {}".format(score))
            
            return pred