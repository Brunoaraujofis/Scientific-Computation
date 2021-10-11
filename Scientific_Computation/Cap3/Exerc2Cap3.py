# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:59:51 2021

@author: Bruno
"""
#CONFIGURAÇÔES 
# Sempre usar no cabeçalho (préambulo) matplotlib e scypi
# Os erros aqui no cabeçalho são devidos ao não uso dos pacotes durante o código,
# um dos erros permite identificar funções onde não especifiquei de qual pacote vem e 
# o outro apenas alerta que o pacote não foi usado. 

from scipy import *
from matplotlib import * 
import numpy as np
import matplotlib.pyplot as plt
import math as mt

# --------------------------- Exercício 3.2 -------------------------------------------------------
m = list(range(100))
print(m)
print(len(m))
L = [k*0.01 for k in m]
for i in list(range(100)):
    L[i] = round(L[i],3)
print(L)
print(len(L))


# Fiz do 0.0 até o 0.99 poderia fazer do 0.0 até 0 1.0 mas ficaria com 101 elementos 
