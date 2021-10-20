# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:37:21 2021

@author: Bruno
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
x = np.arange(0, 3, 0.5)
V = np.vander(x)       
y = np.array([-2.0, 0.5, -2.0, 1.0, -0.5, 1.0])
p, res, rnk, s = sl.lstsq(V, y) # Não e muito claro pra mim porque preciso por p, res, rnk, s = ....
print(np.shape(p))
def poly(z):
    return np.polyval(p, z)
z_valeus = np.linspace(0.0, 2.5, 10)
plt.plot(x,y, '*', label = 'data')
plt.plot(z_valeus, poly(z_valeus), label= "Least Square Fit")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.show()