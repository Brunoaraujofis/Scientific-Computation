
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
import numpy as np
import matplotlib.pyplot as plt
""" Este código inicial define a função recursival """


a = -0.5
h = 1/1000

u = [np.exp(k*h*a) for k in list(range(3)) ]
for n in range(1000):
    last_u = u[n + 2] + h*a*((23/12)*u[n + 2] - (4/3)*u[n + 1] + (5/12)*u[n])
    u.append(last_u)

"""Aqui definiremos a lista td"""
    
td = [n*h for n in range(1003)]

plt.plot(td,u, label = 'Seré recorrente como função de td')

def dif_abs(n):
    return abs(np.exp(a*td[n]) - u[n])
L_difabs = [dif_abs(n) for n in range(1003)]
plt.title("""Plot da serie recorrente u em função
          de td e de diferença absoluta da série recorrente e a exp(td)""")
plt.grid()          
plt.plot(td, L_difabs, label = 'Diferença absoluta da série recorrente e a exp(td)')
plt.legend(loc='center left', fontsize='small')