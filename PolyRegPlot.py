# -*- coding: utf-8 -*-
"""Module DecisionPlot
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator

plt.rcParams["figure.figsize"] = (10,7)


def poly_regr(x, y, degree, extendRight=False):
    x = x[:,np.newaxis]
    
    linear_regression = LinearRegression()

    if degree > 1:
        polynomial_features = PolynomialFeatures(degree=degree)
        pipeline = make_pipeline(polynomial_features, linear_regression)
    else:
        pipeline = linear_regression

    pipeline.fit(x, y)
    
    # für unsere Kennzahlen
    y_pred = pipeline.predict(x)

    # für unseren Regressionsplot
    xmax = np.amax(x) * 1.25 if extendRight else np.amax(x)
    x_test = np.linspace(np.amin(x), xmax, 100)
    y_reg = pipeline.predict(x_test[:, np.newaxis])
    
    # Ein paar Kennzahlen
    rmse = np.sqrt(mean_squared_error(y,y_pred))
    r2 = r2_score(y,y_pred)
    
    print('Polynomial degree={}'.format(degree))
    print('RMSE={}'.format(rmse))
    print('R2  ={}'.format(r2))

    # Plotte unsere Datenpunkte
    plt.scatter(x, y, s=20, label="Samples")
    
    # Plotte die Regressionsfunktion
    plt.plot(x_test, y_reg, color='m', label="Modell")
    
    plt.show()
    
    return pipeline, x, y