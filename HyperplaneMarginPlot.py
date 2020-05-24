import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


def plot_hyperplane_margin(X, y, model, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,10))

    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # create grid to evaluate model
    x1_min, x1_max = X[:, 0].min()  - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min()  - 1, X[:, 1].max() + 1
    xm1, xm2 = np.meshgrid( np.arange(x1_min, x1_max, 0.01),
                            np.arange(x2_min, x2_max, 0.01))
    mesh_points = np.array([xm1.ravel(), xm2.ravel()]).T
    
    Z = model.decision_function(mesh_points).reshape(xm1.shape)

    # plot decision boundary and margins
    ax.contour(xm1, xm2, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

    return ax
                           