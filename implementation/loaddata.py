# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:17:31 2020

@author: PARK
"""
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def disease_load(val_rate = 0.25, seed = 42, target_split = True):
    
    # Load Data
    disease_data = pd.read_csv('../dataset/thyroid_disease/thyroid_disease.csv')
    disease_data = disease_data[['Age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'class']]
    disease_data['outlier'] = disease_data['class'].apply(lambda x: 0 if x == 3 else 1)
    disease_data.drop(columns = 'class', inplace = True)
    
    X = disease_data.drop(columns = 'outlier')
    y = disease_data['outlier']
    
    data_size = X.shape[0]
    idx = np.arange(data_size)
    
    split_size = int(val_rate * data_size)

    np.random.seed(seed)
    np.random.shuffle(idx)
    
    tr_idx, val_idx = idx[split_size:], idx[:split_size]
    
    X_train = X.iloc[tr_idx]
    y_train = y.iloc[tr_idx]
    
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    if target_split == True:
        return X_train, y_train, X_val, y_val
    
    elif target_split == False:
        X_train['label'] = y_train
        X_val['label'] = y_val
        
        return X_train, X_val