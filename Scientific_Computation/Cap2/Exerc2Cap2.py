# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:59:51 2021

@author: Bruno
"""
# Exercício 2 do Capítulo 2 sobre convergencia de série, uso de definições de funções, 
# duplo uso de for com condicional e valor absoluto como condicional.
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

for n in range(30):
    for x in np.arange(0,2*pi, 0.01):
        if dif_abs(x, n) > 1.e-12:
            print("Para i = {} e x = {} a série não é convergente".format(n, x))
            break
        else:
            print("Para i = {} e x = {} a série é convergente".format(n, x))

