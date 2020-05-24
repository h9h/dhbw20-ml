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

    x1_min, x1_max = X[:, 0].min()  - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min()  - 1, X[:, 1].max() + 1
    xm1, xm2 = np.meshgrid( np.arange(x1_min, x1_max, 0.01),
                            np.arange(x2_min, x2_max, 0.01))
    mesh_points = np.array([xm1.ravel(), xm2.ravel()]).T
    
    # predicted vals 
    Z = predictor.predict(mesh_points)
    Z = Z.reshape(xm1.shape)

    # plot contour areas 
    ax.contourf(xm1, xm2, Z, alpha=alpha, cmap=plt.cm.Paired)

    # add a scatter plot of the data points 
    if (bscatter): 
        alpha2 = alpha + 0.4 
        if (alpha2 > 1.0 ):
            alpha2 = 1.0
        plot_samples(X,y,alpha2, ax)
        
    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    return ax