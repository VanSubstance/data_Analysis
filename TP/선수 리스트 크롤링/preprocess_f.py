# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 03:21:49 2020

@author: sungh
"""

import pandas as pd

data_f = pd.read_csv('phase3\\' + 'leagues_f.csv')
data_e = pd.read_csv('phase4\\' + 'leagues_e.csv')
data_f = data_f.drop_duplicates()

result = pd.merge(left = data_e, right = data_f, how = 'left', on = 'name', sort = False)
del result['name']

result.to_csv("phase4\\leagues_g.csv", index = False)
