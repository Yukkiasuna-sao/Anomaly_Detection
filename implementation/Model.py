import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.xgbod import XGBOD

from loaddata import disease_load

class OutlierDetection:
    def __init__(self, data_type, target):
        self.scaler = StandardScaler()

        if data_type == 'disease':
            
            train, test = disease_load(target_split = False)

        X_train = train.drop(columns = target)
        X_test = test.drop(columns = target)

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = train[target]
        self.y_test = test[target]

        self.contamination = np.unique(self.y_train, return_counts = True)[1][1] / len(self.y_train)
        print("Contamination Rate : {} %".format(round(self.contamination * 100, 2)))

    def kNN(self,n_neighbors = 5, method = 'largest', eval_metric = 'auc'):
        self.n_neighbors = n_neighbors
        self.method = method

        model = KNN(contamination = self.contamination, n_neighbors = self.n_neighbors, method = self.method)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            print("AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred)))

        elif eval_metric == 'auccracy':
            print("Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))
    
    def LOF(self,n_neighbors = 20, p = 2, eval_metric = 'auc'):
        self.n_neighbors = n_neighbors
        self.p = p

        model = LOF(contamination = self.contamination, n_neighbors = self.n_neighbors, p = self.p)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            print("AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred)))

        elif eval_metric == 'auccracy':
            print("Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))
    
    def OCSVM(self, kernel = 'rbf', nu = 0.5, eval_metric = 'auc'):
        self.kernel = kernel
        self.nu = nu

        model = OCSVM(contamination = self.contamination, kernel = self.kernel, nu = self.nu)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            print("AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred)))

        elif eval_metric == 'auccracy':
            print("Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))
    
    def iForest(self, eval_metric = 'auc'):

        model = IForest(contamination = self.contamination)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            print("AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred)))

        elif eval_metric == 'auccracy':
            print("Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))
    
    def XGBOD(self, max_depth =3, learning_rate = 0.1, n_estimators = 100, eval_metric = 'auc'):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        model = XGBOD(contamination = self.contamination, max_depth = self.max_depth, learning_rate = self.learning_rate, n_estimators = self.n_estimators)
        model.fit(self.X_train, self.y_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            print("AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred)))

        elif eval_metric == 'auccracy':
            print("Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))
    

        
tmp = OutlierDetection(data_type = 'disease', target = 'label')
tmp.XGBOD()

