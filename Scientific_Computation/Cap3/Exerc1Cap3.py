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

# --------------------------- Exercício 3.1 -------------------------------------------------------

""" Definimos a lista L"""

L = [1,2]
L3 = 3*L

print(L3)

# O conteúdo da L3 são 3 listas L
# Tentaremos prever os elementos 
# L3[0] = 1
# L3[-1] = 2
# L3[10] false

print(L3[0])
print(L3[-1])

# Erramos L3[10] este retorna um erro informando que o elemento da lista esta fora do range

# O comando L4 = [k**2 for k in L3] cria uma lista L4 onde os elementos é o quadrado
# dos elementos da lista L3, assim, L4 é uma lista com 6 elementos (range(5))
# e L4[0] = 1, L4[1]= 4 e fica alternando até L4[5]= 4

L4 = [k**2 for k in L3]
print(L4)

L5 = L3 + L4 
# Este comando define a lista L5 como concatenação da lista L3 e L4

print(L5)
