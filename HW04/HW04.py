# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:21:00 2020

@author: sungh
"""

#%% Initiating

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from dateutil.parser import parse
from scipy import stats, polyval
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.model_selection import cross_val_score as cvs


train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', 
                  header = 0, dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

tgt = 'Sales'

train.columns
vals = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']

#%% Conclusion
discards = ['SchoolHoliday', 'StateHoliday', 'Promo', 'Store']
selects = ['Date', 'Customers', 'Open', 'DayOfWeek']
train = train.drop(discards, axis = 1)

newDay = train['DayOfWeek'] != 7
newDay = newDay.astype(int)
train = train.drop(['DayOfWeek'], axis = 1)
train = pd.concat((train, newDay), axis = 1)

condTrain = (train['Date'] < '2015-01-01')
Xtrain = train[condTrain][selects].drop(['Date'], axis = 1).values
ytrain = train[condTrain]['Sales'].values
Xtest = train[condTrain != True][selects].drop(['Date'], axis = 1).values
ytest = train[condTrain != True]['Sales'].values

#%% Cross validation -> Failed
C_s = np.logspace(-10, 0, 10)

logistic = LogisticRegression()

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)
kf = KFold(n_splits = 3, shuffle = True, random_state = 100)

Xtest[0:236380]
ytest[0:236380]

score = cvs(logistic, Xtrain, ytrain, cv = kf)

accs = []
for c in C_s:
    logistic.C = c
    temp = []
    print("C!\t")
    for Ptrain, Ptest in skf.split(Xtest, ytest):
        print("Fit!\t")
        logistic.fit(Xtest[Ptrain], ytest[Ptest])
        temp.append(logistic.score(Xtest[Ptrain], ytest[Ptest]))
    print("Append!\n")
    accs.append(temp)

accs = np.array(accs)
avg = np.mean(accs, axis = 1)
C_s[np.argmax(avg)]

#%% Learning Method: Linear Regression
train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', 
                  header = 0, dtype={'StateHoliday':'str'})
discards = ['SchoolHoliday', 'StateHoliday', 'Promo', 'Store']
selects = ['Date', 'Customers', 'Open', 'DayOfWeek']
train = train.drop(discards, axis = 1)

newDay = train['DayOfWeek'] != 7
newDay = newDay.astype(int)
train = train.drop(['DayOfWeek'], axis = 1)
train = pd.concat((train, newDay), axis = 1)

condTrain = (train['Date'] < '2015-01-01')
Xtrain = train[condTrain][selects].drop(['Date'], axis = 1).values
ytrain = train[condTrain]['Sales'].values
Xtest = train[condTrain != True][selects].drop(['Date'], axis = 1).values
ytest = train[condTrain != True]['Sales'].values

lin1 = LinearRegression()
lin1.fit(Xtrain, ytrain)
lin1.score(Xtrain, ytrain)
y_pred = lin1.predict(Xtest)
(ytrain == lin1.predict(Xtrain))
(ytest == lin1.predict(Xtest))

y_true = ytest

sse = sum((y_true - y_pred) ** 2)
sst = sum((y_true - np.mean(y_true)) ** 2)
ssr = sst - sse

adj_r2_02 = 1 - (sse / sst)

plt.figure(figsize = (36, 4))
plt.scatter(range(len(ytest)), ytest, marker = 'x')
plt.scatter(range(len(ytest)), y_pred, marker = 'x')

plt.figure(figsize = (12, 8))
plt.scatter(Xtest[:, 2], y_pred, marker = '+')
slope, intercept, r_value, p_value, stderr = stats.linregress(Xtest[:, 2], y_pred)
ry = polyval([slope, intercept], Xtest[:, 2])
plt.plot(Xtest[:, 2], ry, 'r')

#%% Logistic Regression -> Failed -> MemoryError
import gc
gc.collect()
train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', 
                  header = 0, dtype={'StateHoliday':'str'})
discards = ['SchoolHoliday', 'StateHoliday', 'Promo', 'Store']
selects = ['Date', 'Customers', 'Open', 'DayOfWeek']
train = train.drop(discards, axis = 1)

newDay = train['DayOfWeek'] != 7
newDay = newDay.astype(int)
train = train.drop(['DayOfWeek'], axis = 1)
train = pd.concat((train, newDay), axis = 1)

condTrain = (train['Date'] < '2015-01-01')
Xtrain = train[condTrain][selects].drop(['Date'], axis = 1).values
ytrain = train[condTrain]['Sales'].values
Xtest = train[condTrain != True][selects].drop(['Date'], axis = 1).values
ytest = train[condTrain != True]['Sales'].values

lin2 = LogisticRegression()
lin2.fit(Xtrain, ytrain)
lin2.score(Xtrain, ytrain)
y_pred = lin1.predict(Xtest)
(ytrain == lin2.predict(Xtrain))
(ytest == lin2.predict(Xtest))

plt.figure(figsize = (36, 4))
plt.scatter(range(len(ytest)), ytest, marker = 'x')
plt.scatter(range(len(ytest)), y_pred, marker = 'x')

plt.figure(figsize = (12, 8))
plt.scatter(Xtest[:, 0], y_pred, marker = '+')
slope, intercept, r_value, p_value, stderr = stats.linregress(Xtest[:, 0], y_pred)
ry = polyval([slope, intercept], Xtest[:, 0])
plt.plot(Xtest[:, 0], ry, 'r')

#%% KNeighborsRegressor
train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', 
                  header = 0, dtype={'StateHoliday':'str'})
discards = ['SchoolHoliday', 'StateHoliday', 'Promo', 'Store']
selects = ['Date', 'Customers', 'Open', 'DayOfWeek']
train = train.drop(discards, axis = 1)

newDay = train['DayOfWeek'] != 7
newDay = newDay.astype(int)
train = train.drop(['DayOfWeek'], axis = 1)
train = pd.concat((train, newDay), axis = 1)

condTrain = (train['Date'] < '2015-01-01')
Xtrain = train[condTrain][selects].drop(['Date'], axis = 1).values
ytrain = train[condTrain]['Sales'].values
Xtest = train[condTrain != True][selects].drop(['Date'], axis = 1).values
ytest = train[condTrain != True]['Sales'].values

lin2 = KNeighborsRegressor(n_neighbors = 3, weights = "distance")
lin2.fit(Xtrain, ytrain)
lin2.score(Xtrain, ytrain)
y_pred = lin2.predict(Xtest)
(ytrain == lin2.predict(Xtrain))
(ytest == lin2.predict(Xtest))

plt.figure(figsize = (36, 4))
plt.scatter(range(len(ytest)), ytest, marker = 'x')
plt.scatter(range(len(ytest)), y_pred, marker = 'x')

plt.figure(figsize = (12, 8))
plt.scatter(Xtest[:, 2], y_pred, marker = '+')
slope, intercept, r_value, p_value, stderr = stats.linregress(Xtest[:, 2], y_pred)
ry = polyval([slope, intercept], Xtest[:, 2])
plt.plot(Xtest[:, 2], ry, 'b')

#%% Time series Analysis -> VAR 
import statsmodels.api as sm
var1 = sm.tsa.VAR(Xtrain)
result1 = var1.fit()
result1.summary()
result1.forecast(result1.model.endog[-1:], 10)

#%% Time series Analysis -> AR
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

#%% Only the univariate case is implemented
#%% 'Date' and 'Sales'
model = AR(Xtrain)
model_fit = model.fit()


#%% Open -> Select
a = []
for date, week in Xtrain.groupby('Open'):
    a.append(week['Sales'])

plt.figure()
plt.boxplot(a)

#%% Promo -> Discard
train['Promo'].unique
train.groupby('Promo')['Sales'].var()
means = train.groupby('Promo')['Sales'].mean()
std = train.groupby('Promo')['Sales'].std()
plt.bar(range(len(means)), means)
plt.errorbar(range(len(means)), means, yerr = std, fmt = 'o', c = 'r', ecolor = 'r', 
            capthick = 2, capsize = 10)
plt.xticks(range(len(means)), means.index)

train[['Promo', 'Sales']].corr()
plt.figure(figsize = (12, 8))
plt.scatter(train['Promo'], train['Sales'], marker = '+')
slope, intercept, r_value, p_value, stderr = stats.linregress(train['Promo'], train['Sales'])
ry = polyval([slope, intercept], train['Promo'])
plt.plot(train['Promo'], ry, 'r')

a = []
for date, week in Xtrain.groupby('Promo'):
    a.append(week['Sales'])

plt.figure()
plt.boxplot(a)

#%% Customers -> Select

train[['Customers', 'Sales']].corr()
plt.figure(figsize = (12, 8))
plt.scatter(train['DayOfWeek'], train['Sales'], marker = '+')
slope, intercept, r_value, p_value, stderr = stats.linregress(train['DayOfWeek'], train['Sales'])
ry = polyval([slope, intercept], train['DayOfWeek'])
plt.plot(train['DayOfWeek'], ry, 'y')

#%% DayOfWeek -> Select
test = ['DayOfWeek']
train.groupby('DayOfWeek')['Sales'].describe()

a = []
means = [0]
for date, week in Xtrain.groupby('DayOfWeek'):
    a.append(week['Sales'])
    means.append(week['Sales'].mean())

plt.figure()
plt.boxplot(a)
plt.plot(means)
plt.show()

means = train.groupby('DayOfWeek')['Sales'].mean()
std = train.groupby('DayOfWeek')['Sales'].std()
plt.bar(range(len(means)), means)
plt.errorbar(range(len(means)), means, yerr = std, fmt = 'o', c = 'r', ecolor = 'r', 
            capthick = 2, capsize = 10)
plt.xticks(range(len(means)), means.index)

#%% State Holiday -> Discard
means = train.groupby('StateHoliday')['Sales'].mean()
std = train.groupby('StateHoliday')['Sales'].std()
plt.bar(range(len(means)), means)
plt.errorbar(range(len(means)), means, yerr = std, fmt = 'o', c = 'r', ecolor = 'r', 
            capthick = 2, capsize = 10)
plt.xticks(range(len(means)), means.index)

## 실행
train['StateHoliday'].unique
holiday = (train['StateHoliday'] == "0") | (train['StateHoliday'] == 0)
holiday = holiday.astype(int)
train = train.drop(['StateHoliday'], axis = 1)
train = pd.concat((train, holiday), axis = 1)
#### 여기까지


#%% Correlation Graph
corr = train.corr()
fig=plt.figure(figsize=(12,8)) 
cax=plt.imshow(corr, vmin=-1, vmax=1, cmap=plt.cm.RdBu) 
ax=plt.gca() 
ax.set_xticks(range(len(corr))) 
ax.set_yticks(range(len(corr))) 
ax.set_xticklabels(corr,fontsize=10,rotation='vertical') 
ax.set_yticklabels(corr,fontsize=10) 
plt.colorbar(cax)

train[['StateHoliday', 'Sales']].corr()

train[train['Open'] == 1]['Sales'].describe()
train[(train['Open'] == 1) & (train['Sales'] > 8360)].count()

means = train.groupby('Open')['Sales'].mean()
std = train.groupby('Open')['Sales'].std()
plt.bar(range(len(means)), means)
plt.errorbar(range(len(means)), means, yerr = std, fmt = 'o', c = 'r', ecolor = 'r', 
            capthick = 2, capsize = 10)
plt.xticks(range(len(means)), means.index)

train[train['Open'] == 1]
plt.figure()
plt.boxplot(train[train['Open'] == 1]['Sales'])



#%% School Holiday -> Discard
means = train.groupby('SchoolHoliday')['Sales'].mean()
std = train.groupby('SchoolHoliday')['Sales'].std()
plt.bar(range(len(means)), means)
plt.errorbar(range(len(means)), means, yerr = std, fmt = 'o', c = 'r', ecolor = 'r', 
            capthick = 2, capsize = 10)
plt.xticks(range(len(means)), means.index)



"""
plt.plot_date(train['Date'], train['Sales'])
plt.figure(figsize = (20, 1))
plt.plot(train['Date'], train['Sales'], linewidth = 1)
"""

