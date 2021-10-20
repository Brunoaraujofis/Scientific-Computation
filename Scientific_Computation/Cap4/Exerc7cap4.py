# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:25:12 2021

@author: Bruno
"""

import numpy as np

v = np.array([[1, -1, 1]])

P = np.dot(v, np.transpose(v))/ (np.dot(np.transpose(v),v))
print(np.dot(P,np.transpose(v)))
Q = np.identity(3) - P
print(np.dot(Q,np.transpose(v)))

