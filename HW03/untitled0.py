# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:21:00 2020

@author: sungh
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets as ds
from sklearn.neighbors import KNeighborsRegressor


diabets = ds.load_diabetes()
X = diabets.data
y = diabets.target
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
