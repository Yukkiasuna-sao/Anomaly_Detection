""" Isolation Forest Vizualization """

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensembel import IsolationForest

""" Single Feature """
def Vizif(data):    
    fig, axs = plt.subplot(5, 7, figsize = (25, 30), facecolor = 'w', edgecolor = 'k')
    axs = axs.ravel()
    
    for i, column in enumerate(data.columns):
        isolation_forest = IsolationForest()
        isolation_forest.fit(data[column].values.reshape(-1,1))
        
        x_axs = np.linspace(data[column].min(), data[column].max(), len(data)).reshape(-1,1)
        anomaly_score = isolation_forest.decision_function(x_axs)
        outlier = isolation_forest.predict(x_axs)
        
        axs[i].plot(x_axs, anomaly_score, label = 'Anomaly Score')
        axs[i].fill_between(x_axs.T[0], np.min(anomaly_score), np.max(anomaly_score),
                            where = outlier == -1, color = 'r',
                            alpha = .4, label = 'Outlier Region')
        
        axs[i].legend()
        axs[i].set_title(column)
        

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data, get_outliers_inliers

def pyod_vis(method, outliers_fraction = 0.1):
    xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    X_train, Y_train = generate_data(n_train = 300, train_only = True, n_features = 2, random_state = 42)
    
    outlier_fraction = outliers_fraction
    
    x_outliers, x_inliers = get_outliers_inliers(X_train, Y_train)
    n_inliers = len(x_inliers)
    n_outliers = len(x_outliers)
    
    if method == "ABOD":
        clf_name = 'Angle-based Outlier Detector (ABOD)'
        clf = ABOD(contamination = outlier_fraction)
        
        plt.figure(figsize = (10,10))
        
        clf.fit(X_train)
        
        score_pred = clf.decision_function(X_train) * -1
        
        y_pred = clf.predict(X_train)
        
        n_errors = (y_pred != Y_train).sum()
        print('Num of Errors :', clf_name, n_errors)
        
        threshold = stats.scoreatpercentile(score_pred, 100 * outlier_fraction)
        
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        z = z.reshape(xx.shape)
        
        subplot = plt.subplot(1, 1, 1)
         
        subplot.contourf(xx, yy, z, levels = np.linspace(z.min(), threshold, 10), cmap = plt.cm.Blues_r)
    
        a = subplot.contour(xx, yy, z, levels = [threshold], linewidths = 2, colors = 'red')
        subplot.contourf(xx, yy, z, levels = [threshold, z.max()], colors = 'orange')
        b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c = 'white', s = 20, edgecolor = 'k')
        c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c = 'black', s = 20, edgecolor = 'k')
        
        subplot.axis('tight')
        
        subplot.legend([a.collections[0], b, c],
                   ['Learned Decision Function', 'True Inliers', 'True Outliers'])
        subplot.set_title(clf_name)
        subplot.set_xlim((-10, 10))
        subplot.set_ylim((-10, 10))
        
        plt.show()

    
    elif method == 'KNN':
        clf_name = 'K Nearest Neighbors'
        clf = KNN(contamination = outlier_fraction)
    
        plt.figure(figsize = (10,10))
        
        clf.fit(X_train)
        
        score_pred = clf.decision_function(X_train) * -1
        
        y_pred = clf.predict(X_train)
        
        n_errors = (y_pred != Y_train).sum()
        print('Num of Errors :', clf_name, n_errors)
        
        threshold = stats.scoreatpercentile(score_pred, 100 * outlier_fraction)
        
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        z = z.reshape(xx.shape)
        
        subplot = plt.subplot(1, 1, 1)
         
        subplot.contourf(xx, yy, z, levels = np.linspace(z.min(), threshold, 10), cmap = plt.cm.Blues_r)
    
        a = subplot.contour(xx, yy, z, levels = [threshold], linewidths = 2, colors = 'red')
        subplot.contourf(xx, yy, z, levels = [threshold, z.max()], colors = 'orange')
        b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c = 'white', s = 20, edgecolor = 'k')
        c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c = 'black', s = 20, edgecolor = 'k')
        
        subplot.axis('tight')
        
        subplot.legend([a.collections[0], b, c],
                   ['Learned Decision Function', 'True Inliers', 'True Outliers'])
        subplot.set_title(clf_name)
        subplot.set_xlim((-10, 10))
        subplot.set_ylim((-10, 10))
        
        plt.show()
        
    
    elif method == 'IForest':
        clf_name = 'Isolation Forest'
        clf = IForest(contamination = outlier_fraction, random_state = 42)
        
        plt.figure(figsize = (10,10))
        
        clf.fit(X_train)
        
        score_pred = clf.decision_function(X_train) * -1
        
        y_pred = clf.predict(X_train)
        
        n_errors = (y_pred != Y_train).sum()
        print('Num of Errors :', clf_name, n_errors)
        
        threshold = stats.scoreatpercentile(score_pred, 100 * outlier_fraction)
        
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        z = z.reshape(xx.shape)
        
        subplot = plt.subplot(1, 1, 1)
         
        subplot.contourf(xx, yy, z, levels = np.linspace(z.min(), threshold, 10), cmap = plt.cm.Blues_r)
    
        a = subplot.contour(xx, yy, z, levels = [threshold], linewidths = 2, colors = 'red')
        subplot.contourf(xx, yy, z, levels = [threshold, z.max()], colors = 'orange')
        b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c = 'white', s = 20, edgecolor = 'k')
        c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c = 'black', s = 20, edgecolor = 'k')
        
        subplot.axis('tight')
        
        subplot.legend([a.collections[0], b, c],
                   ['Learned Decision Function', 'True Inliers', 'True Outliers'])
        subplot.set_title(clf_name)
        subplot.set_xlim((-10, 10))
        subplot.set_ylim((-10, 10))
        
        plt.show()