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