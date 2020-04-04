# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:48:15 2020

@author: PARK
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def visualization(data, target, vis_type, transform = None):
    subplot_x_axis = data.shape[1] / 2
    subplot_y_axis = data.shape[1] / 3
    if vis_type == 'hist':
 
        f = plt.figure(figsize = (20, 15))
            
        for i, col in enumerate(data.drop(columns = target).columns):
            
            f.add_subplot(subplot_x_axis, subplot_y_axis, i + 1)
                
            if transform == None:
                sns.distplot(data.loc[data[target] == 0][col], label = 'Normal')
                sns.distplot(data.loc[data[target] == 1][col], label = 'Abnormal')
                    
            elif transform == 'log':
                sns.distplot(data.loc[data[target] == 0][col], hist_kws = {'log' : True}, label = 'Normal')
                sns.distplot(data.loc[data[target] == 1][col], hist_kws = {'log' : True}, label = 'Normal')
                
            plt.title(col, fontsize = 14)
            plt.legend(loc = 'best')
            
    elif vis_type == 'box':
        
        f = plt.figure(figsize = (20,15))
        
        for i, col in enumerate(data.drop(columns = target).columns):
            
            f.add_subplot(subplot_x_axis, subplot_y_axis, i + 1)
            
            if transform == None:
                sns.boxplot(data.loc[data[target] == 0][col], label = 'Normal')
                sns.boxplot(data.loc[data[target] == 1][col], label = 'Abnormal')
            
            elif transform == 'log':
                ax = sns.boxplot(x = target, y = col, data = data)
                ax.set_yscale("log")
                ax.set_ylabel("log({})".format(col))
            
            plt.title(col, fontsize = 14)
            plt.legend(loc = 'best')

    elif vis_type == "corr":

        corr_normal = data.loc[data[target] == 0].drop(columns = target).corr()
        corr_abnormal = data.loc[data[target] == 1].drop(columns = target).corr()
    
        cmap = sns.diverging_palette(220, 10, as_cmap = True)
        
        f = plt.figure(figsize = (10,6))
        
        normal = f.add_subplot(1, 2, 1)
        abnormal = f.add_subplot(1, 2, 2)
        
        normal.sns.heatmap(corr_normal, cmap = cmap, vmin = -1, vmax = 1, center = 0,
                           sqaure = True, linewidth = .5, cbar_kws = {'shrink' : .5}, annot = True)
        abnormal.sns.heatmap(corr_abnormal, cmap = cmap, vmin = -1, vmax = 1, center = 0,
                             sqaure = True, linewidth = .5, cbar_kws = {'shrink': .5}, annot = True)
        
        normal.set_title("Correlation Coefficient of Normal")
        abnormal.set_title("Correlation Coefficient of Abnoraml")

        plt.show()        