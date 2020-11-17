# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:24:00 2020

@author: sungh
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import math

data = pd.read_csv('phase4\\' + 'leagues_g.csv')

#%% Changing year into digit 
def year_digit(x):
    if (x > 20):
        x = "19" + str(x)
    elif (0 <= x and x < 10):
        x = "200" + str(x)
    else:
        x = "20" + str(x)
    return int(x)
data['season'] = data['season'].apply(year_digit)

#%%

cols = data.keys()
corr = data[cols].corr()

colsage = sorted(data['age'].unique())

listages = []
for age in colsage:
    listages.append(data[data["age"] == age]["mv"].mean())
    
newOne = pd.DataFrame({"age": colsage, "mv": listages})
def ab(x):
    return abs(x - 28)
newOne['age'] = newOne['age'].apply(ab)
data.boxplot(column=['mv'], by='apearance')

def log(x):
    return math.log(x)
data['mv'] = data['mv'].apply(log)
data['age'] = data['age'].apply(ab)

#%%
catvar=['foot','position']
for c in catvar:
    dummy = pd.get_dummies(data[c], prefix=c, drop_first=True)
    data = pd.concat((data, dummy), axis=1) 
    # combine row-wise
    
for i, x in data.iterrows():
    if x['position'][0] == "F":
        result = x['assist'] + (x['goal'] * 2)
    elif x['position'][0] == "M":
        result = x['goal'] + (x['assist'] * 2)
    else:
        result = x['goal']  + x['assist']
    data.loc[i, 'goal'] = result * x['potential']

#%%
    
data['mv'] = data['mv'] / data['potential']

X = data.drop(catvar+['mv','height'], axis=1)
y = data['mv']

data['potential'] = (data['potential'] - data['potential'].mean())/data['potential'].std()
data['goal'] = data['goal'] * data['potential']

