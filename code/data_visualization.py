# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:02:21 2021

@author: Jerry
"""
import matplotlib.pyplot as plt

#
def plot_true_and_pred(true_value, predicted_value, title, imgpath):
    plt.figure(figsize=(10,10))
    plt.scatter(true_value, predicted_value, c='crimson')
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.title(title,fontsize=18)
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.savefig(imgpath)
