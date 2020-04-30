#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.xgbod import XGBOD

from loaddata import disease_load, tree_load

class OutlierDetection:
    def __init__(self, data_type, target):
        self.scaler = StandardScaler()

        if data_type == 'disease':
            
            train, test = disease_load(target_split = False)

        elif data_type == 'forest':

            train, test = tree_load(target_split = False)

        X_train = train.drop(columns = target)
        X_test = test.drop(columns = target)

        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = train[target]
        self.y_test = test[target]

        self.contamination = np.unique(self.y_train, return_counts = True)[1][1] / len(self.y_train)
        print("Contamination Rate : {} %".format(round(self.contamination * 100, 2)))

    def kNN(self,n_neighbors = 5, method = 'largest', eval_metric = 'auc', return_type = 'metric'):
        self.n_neighbors = n_neighbors
        self.method = method

        model = KNN(contamination = self.contamination, n_neighbors = self.n_neighbors, method = self.method)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]
            print("kNN AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred_proba)))

        elif eval_metric == 'auccracy':
            print("kNN Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))

        elif eval_metric == 'f1':
            precision = precision_score(self.y_test, outlier_pred)
            recall = recall_score(self.y_test, outlier_pred)
            f1 = 2 * (precision * recall) / (precision + recall)

            print("kNN Precision: {}\nRecall: {}\nF1-Score: {}".format(precision, recall, f1))

        if return_type == 'metric':
            return outlier_pred_proba

        elif return_type == 'anoamly':
            train_anomaly_score = model.decision_scores_
            test_anomaly_score =  model.decision_function(self.X_test)

            return train_anomaly_score, test_anomaly_score  

    def LOF(self,n_neighbors = 20, p = 2, eval_metric = 'auc', return_type = 'metric'):
        self.n_neighbors = n_neighbors
        self.p = p

        model = LOF(contamination = self.contamination, n_neighbors = self.n_neighbors, p = self.p)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]
            print("Local Outlier Factor AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred_proba)))

        elif eval_metric == 'auccracy':
            print("Local Outlier Factor Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))

        elif eval_metric == 'f1':
            precision = precision_score(self.y_test, outlier_pred)
            recall = recall_score(self.y_test, outlier_pred)
            f1 = 2 * (precision * recall) / (precision + recall)

            print("Local Outlier Factor\nPrecision: {}\nRecall: {}\nF1-Score: {}".format(precision, recall, f1))

        if return_type == 'metric':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]

            return outlier_pred_proba

        elif return_type == 'anoamly':
            train_anomaly_score = model.decision_scores_
            test_anomaly_score =  model.decision_function(self.X_test)

            return train_anomaly_score, test_anomaly_score 
    
    def OCSVM(self, kernel = 'rbf', nu = 0.5, eval_metric = 'auc', return_type = 'metric'):
        self.kernel = kernel
        self.nu = nu

        model = OCSVM(contamination = self.contamination, kernel = self.kernel, nu = self.nu)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]
            print("OCSVM AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred_proba)))

        elif eval_metric == 'auccracy':
            print("OCSVM Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))

        elif eval_metric == 'f1':
            precision = precision_score(self.y_test, outlier_pred)
            recall = recall_score(self.y_test, outlier_pred)
            f1 = 2 * (precision * recall) / (precision + recall)

            print("OCSVM\nPrecision: {}\nRecall: {}\nF1-Score: {}".format(precision, recall, f1))

        if return_type == 'metric':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]

            return outlier_pred_proba

        elif return_type == 'anoamly':
            train_anomaly_score = model.decision_scores_
            test_anomaly_score =  model.decision_function(self.X_test)

            return train_anomaly_score, test_anomaly_score 
    
    def iForest(self, eval_metric = 'auc', return_type = 'metric'):

        model = IForest(contamination = self.contamination)
        model.fit(self.X_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]
            print("Isolation Forest AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred_proba)))

        elif eval_metric == 'auccracy':
            print("Isolation Forest Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))

        elif eval_metric == 'f1':
            precision = precision_score(self.y_test, outlier_pred)
            recall = recall_score(self.y_test, outlier_pred)
            f1 = 2 * (precision * recall) / (precision + recall)

            print("Isolation Forest\nPrecision: {}\nRecall: {}\nF1-Score: {}".format(precision, recall, f1))

        if return_type == 'metric':
            outlier_pred_proba = model.predict_proba(self.X_test)[::, 1]

            return outlier_pred_proba

        elif return_type == 'anoamly':
            train_anomaly_score = model.decision_scores_
            test_anomaly_score =  model.decision_function(self.X_test)

            return train_anomaly_score, test_anomaly_score 
    
    def XGBOD(self, max_depth =3, learning_rate = 0.1, n_estimators = 100, eval_metric = 'auc', return_type = 'metric'):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        model = XGBOD(contamination = self.contamination, max_depth = self.max_depth, learning_rate = self.learning_rate, n_estimators = self.n_estimators)
        model.fit(self.X_train, self.y_train)

        outlier_pred = model.predict(self.X_test)

        if eval_metric == 'auc':
            outlier_pred_proba = model.predict_proba(self.X_test)
            print("XGBOD AUC Score : {}".format(roc_auc_score(self.y_test, outlier_pred_proba)))

        elif eval_metric == 'auccracy':
            print("XGBOD Accuracy: {}%".format(accuracy_score(self.y_test, outlier_pred) * 100))

        elif eval_metric == 'f1':
            precision = precision_score(self.y_test, outlier_pred)
            recall = recall_score(self.y_test, outlier_pred)
            f1 = 2 * (precision * recall) / (precision + recall)

            print("XGBOD\nPrecision: {}\nRecall: {}\nF1-Score: {}".format(precision, recall, f1))

        if return_type == 'metric':
            outlier_pred_proba = model.predict_proba(self.X_test)

            return outlier_pred_proba

        elif return_type == 'anoamly':
            train_anomaly_score = model.decision_scores_
            test_anomaly_score =  model.decision_function(self.X_test)

            return train_anomaly_score, test_anomaly_score 
     
    def generate_label(self, return_type):
         
        if return_type == 'test':
             return self.y_test

        elif return_type == 'train':
            return self.y_train

def main():
    initializer = OutlierDetection(data_type = 'disease', target = 'label')
    
    eval_df = pd.DataFrame()
    
    eval_df['outlier'] = initializer.generate_label(return_type = 'test')
    eval_df['knn_outlier'] = initializer.kNN()
    eval_df['lof_outlier'] = initializer.LOF()
    eval_df['ocsvm_outlier'] = initializer.OCSVM()
    eval_df['iforest_outlier'] = initializer.iForest()
    eval_df['xgbod_outlier'] = initializer.XGBOD()
    
    def roc_auc_curve(test_data, target, methods = eval_df.columns[1:]):
        plt.figure(figsize = (6, 6))

        for method, i in enumerate(methods):
            fpr, tpr, _ = roc_curve(test_data[target], eval_df[i])
            auc = roc_auc_score(test_data[target], eval_df[i])
            plt.plot(fpr, tpr, label = eval_df.columns[1:][method].split("_")[0].upper() + ", AUC = {}".format(round(auc,2)))
            
            plt.legend(loc = 'best')
            plt.xlabel("False Positive Rate(FPR)")
            plt.ylabel("True Positive Rate(TPR)")

        plt.savefig('./image/comparison.png')
    roc_auc_curve(eval_df, target = 'outlier')

if __name__ == '__main__':
    main()
# %%
