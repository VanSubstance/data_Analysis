# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 03:08:54 2020

@author: sungh
"""

import pandas as pd

data_d = pd.read_csv('phase3\\' + 'leagues_d_final.csv')
data_c = pd.read_csv('phase3\\' + 'leagues_c_with_transfer_no.csv')

result = pd.merge(left = data_c, right = data_d, how = 'inner', on = ['name', 'season'], sort = False)

result.to_csv("phase4\\leagues_e.csv", index = False)
