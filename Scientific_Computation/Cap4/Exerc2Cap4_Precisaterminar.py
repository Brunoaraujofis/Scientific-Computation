# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:44:59 2021

@author: Bruno
"""

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
import scipy.linalg as sl

# --------------------------- Exercício 4.2 -------------------------------------------------------
# Este exercício vamos resolver com uma matriz 3x3 pois a solução com uma matriz 
# 5x5 ficam muito distante devido ao tamanho dos números. Dependendo dos números aleatórios gerados
# A matriz é singular ou a solução não fica muito próxima.
" Queremos construir a matriz de Vandermonde definindo um vetor x aleatório"
x = np.random.randint(1.,5.,(3,)) #retorna um vetor x de 6 entradas inteira entre 1 e 5
print(x)
V = np.array([x**2,x**1, x**0])
V = V.T
print(V)
y = np.random.randint(1., 4., (3,))
print(y)
a = sl.solve(V,y)
print(a)
print(np.allclose(np.dot(V,a),y))


################ Feito sem a escolha aleatória 
# x = np.array([4, 3, 1])
# V = np.array([x**2, x**1, x**0])
# V = V.T
# print(V)
# y = np.array([2, 3, 3])
# a = sl.solve(V,y)
# print(a)
# print(np.allclose(np.dot(V,a),y))
#--------------------------------------------------------------------------------------
def poly(a,z):
    Z = np.array([z**2,z**1, z**0])
    return  np.sum(a*Z)

x = np.arange(0.0,3.0,0.5)
y = np.arange(-2.0,1.5,0.5)
print(x)
