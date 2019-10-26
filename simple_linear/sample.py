# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:25:05 2019

@author: 75100
"""

import numpy
import pandas
import sklearn
import seaborn
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10,8

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression,Ridge,Lasso
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import spline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

regression_data = pandas.read_csv('./data/housing.csv')
regression_data.head()

plt.scatter(regression_data['LSTAT'], regression_data['MEDV'])
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.title('LSTAT - MEDV Data')

x_train, x_test, y_train, y_test = train_test_split(regression_data['LSTAT'], regression_data['MEDV'], test_size=0.20)

simple_linear_regression = LinearRegression()

x_train = x_train.values
y_train = y_train.values
x_test  = x_test.values
simple_linear_regression.fit(pandas.DataFrame(x_train), pandas.DataFrame(y_train))

y_pred = simple_linear_regression.predict(pandas.DataFrame(x_test))

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred, color='red', linewidth=2,)