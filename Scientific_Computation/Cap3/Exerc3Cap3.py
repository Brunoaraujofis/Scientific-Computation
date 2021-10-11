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

L = [0,1,2,0,-1,-2,-1,0]
print(L[0])
print(L[-1])
print(L[: -1])
print( L + L[1:-1] + L)
print(L[2:2] == [-3])
print(L[3:4] == [])
print(L[2:5]== [-5])