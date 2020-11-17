# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:28:31 2020

@author: sungh
"""

import pandas as pd
years = ["16", "17", "18", "19", "20"]
picks = ['short_name', 'international_reputation', 'overall', 'potential', 'physic', 'age',
         'player_positions', 'preferred_foot', 'value_eur']

changes = ['name', 'reputation', 'overall', 'potential', 'physicality', 'age',
         'position', 'foot', 'mv']

for year in years:
    dropper = []
    data = pd.read_csv('data_con\\players_' + year + '.csv')
    for i, player in data.iterrows():
        if player['player_positions'] == "GK":
            dropper.append(i)
    data = data.drop(dropper, axis = 0)
    dataReturn = data[picks]
    dataReturn.columns = changes
    dataReturn.to_csv("result\\newData_" + year + ".csv", index = False)

data = pd.read_csv("result\\newData_16.csv")
k = data.corr()
