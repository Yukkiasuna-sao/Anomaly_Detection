# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:48:15 2020

@author: PARK
"""

import matplotlib.pyplot as plt
import seaborn as sns

def visualization(data, target, vis_type, transform):
    subplot_x_axis = data.shape[1] / 2
    subplot_y_axis = data.shape[1] / 3
    if vis_type == 'distribution':
        
        f = plt.figure(figsize = (20, 15))
        
        for i, col in enumerate(data.drop(columns = target).columns):
            f.add_subplot(subplot_x_axis, subplot_y_axis, i + 1)
            
            sns.distplot(data.loc[data[target] == 0][col], label = 'Normal')
            sns.distplot(data.loc[data[target] == 1][col], label = 'Abnormal')
            
            plt.title(col, fontsize = 14)
            plt.legend(loc = 'best')
            
    elif vis_type == 'box':
        
        f = plt.figure(figsize = (20,15))
        
        for i, col in enumerate(data.drop(columns = target).columns):
            f.add_subplot(subplot_x_axis, subplot_y_axis, i + 1)
            
            sns.boxplot(data.loc[data[target] == 0][col], label = 'Normal')
            sns.boxplot(data.loc[data[target] == 1][col], label = 'Abnormal')
            
            plt.title(col, fontsize = 14)
            plt.legend(loc = 'best')
            