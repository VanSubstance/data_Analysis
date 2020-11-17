# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 01:09:55 2020

@author: sungh
"""

import pandas as pd

data_c = pd.read_csv('phase3\\' + 'leagues_c_final.csv')
    
newOne = pd.Series(list(range(data_c.shape[0])), name = "transfer_no")
result = pd.concat([data_c, newOne], axis = 1)

for player in data_c['name'].unique():
    indexes = data_c[data_c['name'] == player].index
    part = []
    parts = []
    for seq in range(len(indexes)):
        if seq + 1 == len(indexes):
            part.append(indexes[seq])
            parts.append(part)
            part = []
        elif indexes[seq + 1] - indexes[seq] == 1:
            part.append(indexes[seq])
        else:
            part.append(indexes[seq])
            parts.append(part)
            part = []
    
    for par in parts:
        siz = len(par)
        for seq in par:
            data_c.loc[seq, 'transfer_no'] = siz
            siz -= 1
    print(player, " |\n")

data_c.to_csv('phase3\\' + 'leagues_c_with_transfer_no.csv')
