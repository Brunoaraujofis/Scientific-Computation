# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:41:00 2021

@author: Bruno
"""

import numpy as np

""" Não tentarei evitar o uso de loops"""

u = np.random.randint(0, 10, (10,))
chi = np.array([(u[1] + u[i+1] + u[i+2])/3 for i in np.arange(8)])
print(u)
print(chi)


