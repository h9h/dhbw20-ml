# -*- coding: utf-8 -*-
"""Module DecisionPlot
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker, cm

from matplotlib.colors import ListedColormap

def plot_decision_surface(X, y, predictor, ax_delta=1.0, mesh_res = 0.01, alpha=0.4, bscatter=True,  
                          x1_label='x1', x2_label='x2', legend_loc='upper right'):

    # some arrays and colormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # mesh points 
    resolution = mesh_res
    x1_min, x1_max = X[:, 0].min()  - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min()  - 1, X[:, 1].max() + 1
    xm1, xm2 = np.meshgrid( np.arange(x1_min, x1_max, resolution), 
                            np.arange(x2_min, x2_max, resolution))
    mesh_points = np.array([xm1.ravel(), xm2.ravel()]).T

    # predicted vals 
    Z = predictor.predict(mesh_points)
    Z = Z.reshape(xm1.shape)

    # plot contur areas 
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.contourf(xm1, xm2, Z, alpha=alpha, cmap=cmap)

    # add a scatter plot of the data points 
    if (bscatter): 
        alpha2 = alpha + 0.4 
        if (alpha2 > 1.0 ):
            alpha2 = 1.0
        for idx, yv in enumerate(np.unique(y)): 
            ax.scatter(x=X[y==yv, 0], y=X[y==yv, 1], 
                        alpha=alpha2, c=[cmap(idx)], marker=markers[idx], label=yv)
            
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    if (bscatter):
        ax.legend(loc=legend_loc)