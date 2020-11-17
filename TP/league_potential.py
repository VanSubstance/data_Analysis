# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:42:53 2020

@author: sungh
"""

import pandas as pd

leagues = ["bundesliga", "campeonato-brasileiro-serie-a", "campeonato-uruguayo-copa-coca-cola", "eredivisie", 
           "italian-serie-a", "liga-nos", "ligue-1-conforama", "spanish-first-division", "super-liga-argentina"]



data = pd.read_csv("phase2\\" + "leagues_bc" + ".csv", error_bad_lines=False)