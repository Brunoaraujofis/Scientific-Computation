# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:01:46 2021

@author: Bruno
"""

#CONFIGURAÇÔES 
# Sempre usar no cabeçalho (préambulo) matplotlib e scypi
# Os erros aqui no cabeçalho são devidos ao não uso dos pacotes durante o código,
# um dos erros permite identificar funções onde não especifiquei de qual pacote vem e 
# o outro apenas alerta que o pacote não foi usado. 

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl

x = np.arange(0, 3, 0.5)
y = np.array([-2.0, 0.5, -2.0, 1.0, -0.5, 1.0])

V = np.vander(x)
a = sl.solve(V,y)
print(np.shape(a))
def Poly(x):
    p = np.polyval(a,x)
    return p

xvals = np.linspace(0.0, 2.5, 10)

plt.plot(x, y, '*', label = 'data')
plt.plot(xvals, Poly(xvals), label = 'Least Square Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(framealpha = 1, shadow= True)
plt.grid(alpha=0.5)
plt.show()