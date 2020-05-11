import os
import pandas as pd
import numpy as np
import scipy.io as scio

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier # BaggingClassifier based on Undersampling
from PyNomaly import loop # Local Outlier Probability: https://github.com/vc1492a/PyNomaly

from models.knn import Knn
from models.utility import get_precn, print_baseline

import warnings
warnings.filterwarnings('ignore')

def loaddata(data):
    if data == 'letter':
        mat = scio.loadmat(os.path.join('datasets', 'letter.mat'))

    X = mat['X']
    y = mat['y']

    return X, y

class XGBOD:
    def __init__(self, test_size, select_TOS = 'random', num_TOS = 10, random_seed = 42):
        self.test_size = test_size
        self.num_TOS = num_TOS
        self.select_TOS = select_TOS
        self.random_seed = random_seed
        
    def fit(self, X, y, lofp = False):
        self.X = X
        self.y = y

        self.out_perc = np.count_nonzero(self.y) / len(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
        
        # Data Normalization
        scaler = Normalizer()
        self.X_train_norm = scaler.fit_transform(self.X_train)
        self.X_test_norm = scaler.transform(self.X_test)

        feature_list = []

        # Predefined Range of K (Using Knn)
        k_list_pre = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]
    
        # Trim the list in case of small sample size
        ## Example) sample size = 200   k size = 250 -> Error
        k_list = [k for k in k_list_pre if k < self.X_train.shape[0]]

        ##################################### KNN Largest Method ################################################
        train_knn = np.zeros([self.X_train.shape[0], len(k_list)])
        test_knn = np.zeros([self.X_test.shape[0], len(k_list)])

        for i in range(len(k_list)):
            k = k_list[i]

            print("kNN (Largest) Algorithm Modeling.. k = {}".format(k))
            clf = Knn(n_neighbors = k, contamination = self.out_perc, method = 'largest')
            clf.fit(self.X_train_norm)
            train_score = clf.decision_scores
            pred_score = clf.decision_function(self.X_test_norm)

            feature_list.append('knn_' + str(k))

            train_knn[:, i] = train_score
            test_knn[:, i] = pred_score.ravel()
        
        ##################################### KNN Mean Method ################################################
        train_knn_mean = np.zeros([self.X_train.shape[0], len(k_list)])
        test_knn_mean = np.zeros([self.X_test.shape[0], len(k_list)])

        for i in range(len(k_list)):
            k = k_list[i]

            print("kNN (Mean) Algorithm Modeling.. k = {}".format(k))
            clf = Knn(n_neighbors = k, contamination = self.out_perc, method = 'mean')
            clf.fit(self.X_train_norm)
            train_score = clf.decision_scores
            pred_score = clf.decision_function(self.X_test_norm)

            feature_list.append('knn_mean_' + str(k))

            train_knn_mean[:, i] = train_score
            test_knn_mean[:, i] = pred_score.ravel()
       
        ##################################### KNN Median Method ################################################
        train_knn_median = np.zeros([self.X_train.shape[0], len(k_list)])
        test_knn_median = np.zeros([self.X_test.shape[0], len(k_list)])

        for i in range(len(k_list)):
            k = k_list[i]

            print("kNN (Median) Algorithm Modeling.. k = {}".format(k))
            clf = Knn(n_neighbors = k, contamination = self.out_perc, method = 'median')
            clf.fit(self.X_train)
            train_score = clf.decision_scores
            pred_score = clf.decision_function(self.X_test_norm)

            feature_list.append('knn_median_' + str(k))
            
            train_knn_median[:, i] = train_score
            test_knn_median[:, i] = pred_score.ravel()
        
        ##################################### Local Outlier Factor ################################################
        train_lof = np.zeros([self.X_train.shape[0], len(k_list)])
        test_lof = np.zeros([self.X_test.shape[0], len(k_list)])

        for i in range(len(k_list)):
                k = k_list[i]
                print("Local Outlier Factor Algorithm Modeling.. k = {}".format(k))
                clf = LocalOutlierFactor(n_neighbors=k)
                clf.fit(self.X_train_norm)
                # save the train sets
                train_score = clf.negative_outlier_factor_ * -1
                # flip the score
                pred_score = clf._decision_function(self.X_test_norm) * -1

                feature_list.append('lof_' + str(k))

                train_lof[:, i] = train_score
                test_lof[:, i] = pred_score

        ##################################### Local Outlier Probabilites ################################################
        # LoOP는 모델 복잡도가 매우 높기 때문에 실제 예측에 사용되지 않는다.
        # 여기에선 XGBOD의 효과를 입증하기 위해 포함되었다.

        if lofp == True:
            df_X = pd.DataFrame(np.concatenate([X_train_norm, X_test_norm], axis = 0))

            k_list = [1, 5, 10, 20]

            train_loop = np.zeros([self.X_train.shape[0], len(k_list)])
            test_loop = np.zeros([self.X_test.shape[0], len(k_list)])

            for i in range(len(k_list)):
                k = k_list[i]
                print("Local Outlier Factor Probabilites Algorithm Modeling.. k = {}".format(k))
                clf = loop.LocalOutlierProbability(df_X, n_neighbors = k).fit()
                score = clf.local_outlier_probabilities.astype(float)
                train_score = score[:self.X_train.shape[0]]
                pred_score = score[self.X_train.shape[0]:]

                feature_list.append('loop_' + str(k))

                train_loop[:, i] = train_score
                test_loop[:, i] = test_score

        ##################################### One Class Support Vector Machine ################################################
        nu_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

        train_svm = np.zeros([self.X_train.shape[0], len(nu_list)])
        test_svm = np.zeros([self.X_test.shape[0], len(nu_list)])

        for i in range(len(nu_list)):
            nu = nu_list[i]
            print("One Class Support Vector Machine Algorithm Modeling.. Nu = {}".format(nu))
            clf = OneClassSVM(nu = nu)
            clf.fit(self.X_train_norm)

            train_score = clf.decision_function(self.X_train_norm) * -1
            pred_score = clf.decision_function(self.X_test_norm) * -1

            feature_list.append('ocsvm_' + str(nu))

            train_svm[:, i] = train_score.ravel()
            test_svm[:, i] = pred_score.ravel()
        ##################################### Isolation Forest ################################################
        n_estimators_list = [10, 20, 50, 70, 100, 150, 200, 250]

        train_if = np.zeros([self.X_train.shape[0], len(n_estimators_list)])
        test_if = np.zeros([self.X_test.shape[0], len(n_estimators_list)])

        for i in range(len(n_estimators_list)):
            n = n_estimators_list[i]

            print("Isolation Forest Algorithm Modeling.. n_estimators = {}".format(n))
            clf = IsolationForest(n_estimators = n)
            clf.fit(self.X_train)
            
            train_score = clf.decision_function(self.X_train) * -1
            pred_score = clf.decision_function(self.X_test) * -1

            feature_list.append('if_' + str(n))

            train_if[:, i] = train_score
            test_if[:, i] = pred_score
        
        ##################################### Merged Dataset ################################################
        if lofp == True:

            X_train_TOS = np.concatenate((train_knn, train_knn_mean, train_knn_median, train_lof, train_loop, train_svm, train_if), axis = 1)
            X_test_TOS = np.concatenate((test_knn, test_knn_mean, test_knn_median, test_lof, test_loop, test_svm, test_if), axis = 1)

        else:

            X_train_TOS = np.concatenate((train_knn, train_knn_mean, train_knn_median, train_lof, train_svm, train_if), axis = 1)
            X_test_TOS = np.concatenate((test_knn, test_knn_mean, test_knn_median, test_lof, test_svm, test_if), axis = 1)
        #X_train_all = np.concatenate((X_train, X_train_TOS), axis = 1)
        #X_test_all = np.concatenate((X_test, X_test_TOS), axis = 1)

        ##################################### Select TOS Using Difference Method ################################################
        print("Select TOS P = {}".format(self.num_TOS))
        p = self.num_TOS # Number of Selected TOS

        # Random Select Method
        if self.select_TOS == 'random':
            indice = np.arange(p)
            
            np.random.seed(self.random_seed)
            np.random.shuffle(indice)

            X_train_TOS = X_train_TOS[:, indice]
            X_test_TOS = X_test_TOS[:, indice]

        self.X_train_all = np.concatenate((self.X_train, X_train_TOS), axis = 1)
        self.X_test_all = np.concatenate((self.X_test, X_test_TOS), axis = 1)

        self.clf = XGBClassifier(random_state = self.random_seed)
        self.clf.fit(self.X_train_all, self.y_train.ravel())

        return self.clf
    
    def predict(self):
        y_pred = self.clf.predict(self.X_test_all)

        return y_pred
    
    def predict_proba(self):
        y_pred = self.clf.predict_proba(self.X_test_all)

        #roc_score = roc_auc_score(y_pred[:, 1], self.y_test.ravel())
        #precision_score = get_precn(y_pred[:, 1], self.y_test.ravel())
        #print(y_pred[:, 1].shape)
        #print(self.y_test.ravel().shape)
        #print('AUC :', round(roc_score, 2))
        #print('Precision : ', round(precision_score, 2))

        return y_pred[:, 1]

def main():
    X, y = loaddata(data = 'letter')
    
    model = XGBOD(test_size = 0.3, select_TOS = 'random', num_TOS = 5, random_seed = 42)
    model.fit(X,y)

    pred = model.predict_proba()

main()