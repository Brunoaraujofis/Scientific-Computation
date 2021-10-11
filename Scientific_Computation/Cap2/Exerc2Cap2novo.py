# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:47:47 2021

@author: Bruno
"""

from scipy import *
from matplotlib import * 
import numpy as np
import matplotlib.pyplot as plt
import math as mt

def f(x,n):
    return (mt.cos(x) + 1J*mt.sin(x))**n

def g(x,n):
    return mt.cos(n*x) + 1j*mt.sin(n*x)

def dif_abs(x,n):
    return abs(f(x,n) - g(x,n))

L = []

for n in range(30):
    for x in np.arange(0,2*pi,0.1):
        L.append(dif_abs(x, n))     
        

for i in range(len(L)):
    if L[i] < 1.e-8:
        L[i] = 0.0     
for i in range(len(L)):
    if L[i] != 0.0:
        print("A série não é convergente")
        break
    if i == len(L) - 1 and L[i] == 0.0:
        print("A série é convergente")
            
