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


""" Queremos definir uma função que calcule a diferença simétrica entre dois conjuntos"""

A = {0,1,2,3}
B = {'a', 'b','c', 1,2}

D = A.difference(B)
E = B.difference(A)
F = D.union(E)

print(F)

print(A.symmetric_difference(B))


"O comando A.symmetric_difference(B) é equivalente a função definida"