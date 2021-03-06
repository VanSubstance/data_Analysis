# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 00:01:23 2020

@author: sungh
"""


import pandas as pd
import matplotlib.pyplot as plt

house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')
house
house.keys()

#%% EDA

cols = house.keys()
corr = house[cols].corr()

# 1. Remove 'price' (which is the one we are going to treat as 'y') from cols
y = 'price'
cols = list(cols)
cols.remove('price')

# 2. divide cols into 4 pieces to see scatter matrix by each
# It is too big for using at once
cols1 = cols[0:5]
cols2 = cols[5:10]
cols3 = cols[10:15]
cols4 = cols[15:]

# 3. Scatter matrix for each cols's particles and 'y'
cols1.append(y)
cols2.append(y)
cols3.append(y)
cols4.append(y)
from pandas.plotting import scatter_matrix
scatter_matrix(house[cols1], figsize = (12, 8))
scatter_matrix(house[cols2], figsize = (12, 8))
scatter_matrix(house[cols3], figsize = (12, 8))
scatter_matrix(house[cols4], figsize = (12, 8))

# 4. Transform value 'date' into integer which is the form we can use

house['date']
for ind in range(len(house['date'])):
    newDate = house['date'][ind][:8]
    """
    if int(newDate[:4]) == 2014:
        newDate = int(newDate[4:])
    else:
        newDate = int('1' + newDate[4:])
    """
    house['date'][ind] = newDate
house['date'] = pd.to_numeric(house['date'])

scatter_matrix(house[['date', 'price']], figsize = (12, 8))

house[['date', 'price']].corr()

# 5. Try to transform sqft_lot, sqft_lot15 into a shape that has straight correlation with y(price)
# Failed

import math

max1 = house['sqft_lot'].max()
max2 = house['sqft_lot15'].max()
for ind in range(len(house)):
    newData = house['sqft_lot'][ind]
    newData2 = house['sqft_lot15'][ind]
    house['sqft_lot'][ind] = math.sqrt(max1 - newData)
    house['sqft_lot15'][ind] = math.sqrt(max2 - newData2)

plt.scatter(house['price'], house['sqft_lot'])

house.groupby('price')['sqft_lot'].count().plot()

scatter_matrix(house[['sqft_lot', 'sqft_lot15', 'price']], figsize = (12, 8))
scatter_matrix(house[['sqft_living', 'sqft_living15', 'price']], figsize = (12, 8))

#%% Linear regression

import statsmodels.api as sm
from sklearn import datasets

X = house[['bathrooms', 'sqft_living', 'grade', 'sqft_above', 
           'sqft_basement', 'sqft_living15']]

y = house['price']

X = sm.add_constant(X)

model = sm.OLS(y, X)
result = model.fit()

result.summary()


