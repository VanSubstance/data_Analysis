# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:28:20 2020

@author: sungh
"""

import pandas as pd

#%% C 시즌별로 mv 평균 내서 정리

data_c = pd.read_csv('phase3\\' + 'leagues_c.csv')

indexCol = []
for name in data_c['name'].unique():
    for age in data_c[data_c['name'] == name]['age'].unique():
        for date in data_c[data_c['name'] == name][data_c['age'] == age]['date'].unique():
            mv = data_c[data_c['name'] == name][data_c['age'] == age][data_c['date'] == date]['mv'].mean()
            index = data_c[data_c['name'] == name][data_c['age'] == age][data_c['date'] == date]['mv'].index[0]
            data_c.loc[index, 'mv'] = mv
            indexCol.append(index)
            print(name, " | ", date)

data_c.loc[indexCol].to_csv("phase3\\leagues_c_final.csv", index = False)