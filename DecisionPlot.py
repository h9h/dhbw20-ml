# -*- coding: utf-8 -*-
"""Module DecisionPlot
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker, cm

from matplotlib.colors import ListedColormap

def plot_samples(X, y, alpha=0.8, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,10))

    for idx, yv in enumerate(np.unique(y)): 
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired, alpha=alpha)
    return ax
    
def plot_decision_surface(X, y, predictor, ax_delta=1.0, alpha=0.4, bscatter=True,  
                          x1_label='x1', x2_label='x2', legend_loc='upper right', ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,10))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY = np.meshgrid(xx, yy)
    mesh_points = np.vstack([XX.ravel(), YY.ravel()]).T

    # predicted vals 
    Z = predictor.predict(mesh_points)
    Z = Z.reshape(XX.shape)

    # plot contour areas 
    ax.contourf(XX, YY, Z, alpha=alpha, cmap=plt.cm.Paired)

    # add a scatter plot of the data points 
    if (bscatter): 
        alpha2 = alpha + 0.4 
        if (alpha2 > 1.0 ):
            alpha2 = 1.0
        plot_samples(X,y,alpha2, ax)
        
    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    return ax