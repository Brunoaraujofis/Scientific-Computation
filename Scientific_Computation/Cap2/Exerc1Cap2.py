# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:25:03 2021

@author: Bruno
"""

# Exercício 1 do Capítulo 2 sobre convergencia de série, uso de definições de funções, 
# e condicional.

from scipy import *
from matplotlib import * 
import numpy as np
import matplotlib.pyplot as plt
import math as mt

def f(x):
    return x**2 + 0.25*x - 5

def dif_abs(x):
    return abs(0 - f(x))
f(2.3)
if dif_abs(2.3) < 1.e-8:
    print("2.3 é raiz da função f".format(f(2.3)))
    
else:
     print("2.3 não é raiz da função f, pois f(2.3) = {}".format(f(2.3)))