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

# --------------------------- Exercício 4.1 -------------------------------------------------------

M = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])

M1 = np.arange(13)
M1 = M1[1:]
M2 = M1.reshape(np.shape(M))
            
for i in np.arange(4):
     for j in np.arange(3):
         if M[i,j] != M2[i,j]:
             print("As matrizes são diferentes")
             break
         if i == 3 and j ==2 and M[i,j] == M2[i,j]:
             print("As matrizes são identicas")
             
print(M[2,:])
print(M[2:])