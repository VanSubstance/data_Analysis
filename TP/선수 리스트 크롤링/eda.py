# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:24:00 2020

@author: sungh
"""

import pandas as pd

data = pd.read_csv('phase4\\' + 'leagues_g.csv')


cols = data.keys()
corr = data[cols].corr()