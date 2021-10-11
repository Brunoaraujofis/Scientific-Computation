# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:33:15 2021

@author: Bruno
"""

# Exercício 3 do Capítulo 2 sobre convergencia de série, uso de definições de funções, 
# duplo uso de for com condicional e valor absoluto como condicional.
from scipy import *
from matplotlib import * 
import numpy as np
import matplotlib.pyplot as plt
import math as mt

def f(x):
    return mt.cos(x) + 1j*mt.sin(x)

def g(x):
    return mt.e**(1j*x)

def dif_abs(x):
    return abs(f(x) - g(x))

for x in np.arange(0,2*pi, 0.1):
    if dif_abs(x) > 1.e-12:
        print("Para x = {} a série não é convergente".format(x))
        break
    else:
        print("Para x = {} a série é convergente".format(x))
    