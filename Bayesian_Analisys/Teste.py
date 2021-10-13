# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 23:49:49 2021

@author: Bruno
"""

import pandas as pd
gss = pd.read_csv('gss_bayes.csv', index_col=0)
gss.head()