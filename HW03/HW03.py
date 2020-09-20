# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:36:36 2020

@author: sungh
"""
#%% Import section
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import datasets
from pandas.plotting import scatter_matrix

house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

#%% Training with bare setting
X = house[['bathrooms', 'sqft_living', 'grade', 'sqft_above', 
           'sqft_basement', 'sqft_living15']]
y = house['price']
X = sm.add_constant(X)
model = sm.OLS(y, X)
result = model.fit()
result.summary()

#%% Add categorial variables to variable set
X2 = X
varCtgr = ['view', 'condition', 'grade']
for c in varCtgr:
    dummy = pd.get_dummies(house[c], prefix = c, drop_first = True)
    X2 = pd.concat((X2, dummy), axis = 1)

X2 = sm.add_constant(X2)
model2 = sm.OLS(y, X2)
result2 = model2.fit()
result2.summary()

#%% Ideas to utilize zipcode, lat, and long

plt.boxplot(house['view'])
scatter_matrix(house[['zipcode', 'lat', 'long', 'price']])
house['lat'].describe()
house['long'].describe()

scatter_matrix(house[['lat', 'long', 'zipcode', 'price']])
plt.boxplot(house['zipcode'])

plt.scatter(house['lat'], house['long'], marker = '+', s = 1)
plt.scatter(house['lat'].median(), house['long'].median(), marker = 'X', s = 50, c ='r')
reZip = house['zipcode'].apply(lambda x: abs(house['zipcode'].median() - x))
reLat = house['lat'].apply(lambda x: abs(house['lat'].median() - x))
reLong = house['long'].apply(lambda x: abs(house['long'].median() - x))
plt.scatter(reLat, reLong, marker = '+', s = 1)
plt.xlabel("lat - medain of lat")
plt.ylabel("long - medain of long")

X3 = X
XX = pd.concat((house['sqft_living'], house['condition'], reZip, reLat, reLong, y), axis = 1)
scatter_matrix(XX)
