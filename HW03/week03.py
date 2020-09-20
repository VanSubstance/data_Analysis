# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 01:09:32 2020

@author: sungh
"""
#%% 각 그룹별 평균값과 표준편차 시각화
import pandas as pd

salary = pd.read_csv('https://drive.google.com/uc?export=download&id=1kkAZzL8uRSak8gM-0iqMMAFQJTfnyGuh')
salary

dummy = pd.get_dummies(salary['sex'], prefix = 'sex', drop_first = False)

varname = 'rank'
gmean = salary.groupby(varname)['salary'].mean()
gmean

gstd = salary.groupby(varname)['salary'].std()
gstd

import matplotlib.pyplot as plt

plt.bar(range(len(gmean)), gmean)
plt.errorbar(range(len(gmean)), gmean, yerr = gstd, fmt = 'o', c = 'r', ecolor = 'r', 
            capthick = 2, capsize = 10)
plt.xticks(range(len(gmean)), gmean.index)

#%% 분산분석 = ANOVA 검정
from scipy.stats import f_oneway

salary['rank'].value_counts()
groups = [x[1] for x in salary.groupby(['rank'])['salary']]

# P-value가 0.05보다 작다 = 그룹간의 평균이 모두 같지 않다
f_oneway(*groups)

catvar = ['rank', 'discipline', 'sex']
for c in catvar:
    dummy1 = pd.get_dummies(salary[c], prefix = c, drop_first = True)
    salary = pd.concat((salary, dummy1), axis = 1)
    
X = salary.drop(catvar + ['salary'], axis = 1)
y = salary['salary']

#%% 회귀 분석
import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.OLS(y, X)
result = model.fit()

result.summary()

#%% 두번째 실습
house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')
house
house.columns
# price = target


varlist = ['bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'sqft_above', 'sqft_basement', 
           'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

X1 = house[varlist]
X1 = sm.add_constant(X1)
y1 = house['price']

model1 = sm.OLS(y1, X1)
result1 = model1.fit()
result1.summary()

y_pred1 = result1.predict(X1)
err1 = y1 - y_pred1

plt.scatter(y1, y_pred1)
plt.ylabel('Predicted')
plt.xlabel('Real')

import numpy as np

xx = np.linspace(y1.min(), y1.max(), 100)
plt.plot(xx, xx, color = 'k')

plt.hist(err1, bins = 50)

from scipy.stats import probplot
from statsmodels.stats import diagnostic

# 정규분포선을 따르지 않는 부분 확인 가능
probplot(err1, plot = plt)
# 1, 3 = P-value for each test: LM, F
diagnostic.het_breuschpagan(err1, X1)
diagnostic.het_breuschpagan(err1, X1[['bedrooms', 'bathrooms']])


cond = (house['price'] < 1000000)&(house['price'] >= 20000)
X2 = house[cond][varlist]
y2 = house[cond]['price']
X2 = sm.add_constant(X2)
y2.plot.kde()
model2 = sm.OLS(y2, X2)
result2 = model2.fit()
result2.summary()
# R-square is the smallest
# Reason: There is very large outlayers
#%% Mean squared error 계산
from sklearn.metrics import mean_squared_error, mean_absolute_error, 
mean_squared_log_error, median_absolute_error

y_pred2 = result.predict(X2)
mean_squared_error(y1, y_pred1)
mean_squared_error(y2, y_pred2)
mean_squared_error(y1[cond], y_pred1[cond])

# Better to remove outlayer