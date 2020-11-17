# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 01:09:55 2020

@author: sungh
"""

import pandas as pd

data_c = pd.read_csv('phase3\\' + 'leagues_c_final.csv')
    
newOne = pd.Series(list(range(data_c.shape[0])), name = "transfer_no")
result = pd.concat([data_c, newOne], axis = 1)
dropper = []

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
            
    siz = 1
    for par in parts:
        siz = 1
        for seq in range(len(par) - 1, -1, -1):
            data_c.loc[par[seq], 'transfer_no'] = siz
            if seq != 0:
                data_c.loc[par[seq], 'mv'] = data_c.loc[par[seq - 1], 'mv']
            elif seq == 0:
                dropper.append(par[seq])
            siz += 1
    print(player, " |\n")
    
data_c = data_c.drop(dropper, axis = 0)
data_c.to_csv('phase3\\' + 'leagues_c_with_transfer_no.csv', index = False)
