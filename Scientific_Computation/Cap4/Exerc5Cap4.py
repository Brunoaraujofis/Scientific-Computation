# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:20:09 2021

@author: Bruno
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
x = np.arange(0, 3, 0.5)
V = np.vander(x)       
A = np.array(V[:, 1:])
K = np.dot(A.transpose(),A)
K_inverse = sl.inv(K)
B = np.dot(K_inverse, A.transpose())
y = np.array([-2.0, 0.5, -2.0, 1.0, -0.5, 1.0])
c = np.dot(B,y)
print(np.shape(c))
def poly(z):
    return np.polyval(c, z)
z_valeus = np.linspace(0.0, 2.5, 10)
plt.plot(x,y,'*', label = "data")
plt.plot(z_valeus, poly(z_valeus), label = "Least Squares Fit")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.5)
plt.show()