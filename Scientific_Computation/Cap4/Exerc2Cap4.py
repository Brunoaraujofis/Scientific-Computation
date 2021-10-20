# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:44:59 2021

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
# x = np.random.randint(1.,5.,(3,)) #retorna um vetor x de 6 entradas inteira entre 1 e 5
# print(x)
# V = np.array([x**2,x**1, x**0])
# V = V.T
# print(V)
# y = np.random.randint(1., 4., (3,))
# print(y)
# a = sl.solve(V,y)
# print(a)
# print(np.allclose(np.dot(V,a),y))


################ Feito sem a escolha aleatória 
x = np.arange(0, 3, 0.5)
#print(x**5)
y = np.array([-2.0, 0.5, -2.0, 1.0, -0.5, 1.0])
V = np.array([x**(i+5) for i in np.arange(6)])
V = V.T
print(V)
# a = sl.solve(V,y)
#--------------------------------------------------------------------------------------

# Definindo duas formas diferentes de fazer a função Poly

def poly(z):
    Z = np.array([z**i for i in np.arange(6)])
    return  np.sum(a*Z)

def poly1(z):
    p = 0
    for i in np.arange(6):
        p = p + a[5-i]*z**i
    return p
# xvals = np.linspace(0.0, 2.5, 10)
# plt.plot(xvals, poly1(xvals))
# plt.scatter(x, y, marker = "*", norm = True)
# plt.show()

"""Não consigo entender porque os pontos ficam distribuidos daquela forma, possívelmente
é a escala de y, porque a função poly explode muito facilmente... 
A matriz de Vandermonde não foi definida de forma correta."""