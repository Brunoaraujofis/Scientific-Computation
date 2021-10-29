# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:48:31 2021

@author: Bruno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import random

##################### Uniform Distribution ########################################################

""" Usamos stats.distribution.rvs() para gerar pontos aleatórios para uma dada
distribuição, no caso a seguir usamos a distribuição uniform como primeiro exemplo"""

# uniform_data = stats.uniform.rvs(size = 100000, loc = 0, scale = 10)
# pd.DataFrame(uniform_data).plot(kind = 'density', figsize = (9,9), xlim = (-1,11))

""" Usamos stats.distribution.cdf() para determinar a probabilidade de obter um valor abaixo de
um valor limite chamado cutoff"""

# prob_of_measure = stats.uniform.cdf(x =2.5 , loc = 0, scale = 10.0)
# print(prob_of_measure)

""" stats.distribution.ppf() é a função inversa de cdf retornando o valor x do cutoff
associado com uma dada probabilidade."""

# cutoff = stats.uniform.ppf(q = 0.4, loc = 0, scale = 10)
# print(cutoff)

""" stats.distribution.pdf() fornece a densidade de probabilidade (altura da distribuição)
em um valor x """

# for x in np.arange(-1,12,3):
#     print( " Density at x value: " + str(x))
#     print(stats.uniform.pdf(x, loc = 0, scale = 10))

##################### Gerar número aleatórios ########################################################


""" Para gerar números reais aleatórios em um determinado range com igual probabilidade, 
podemos usar stats.uniforme.rvs(). Contúdo, existe uma library "random" que lhe permite 
fazer algumas operações envolvendo randomização"""

"""Quando fazemos algumas operações com a library random obtemos números ditos 
"pseudoaleatórios"  que serão comentados a seguir"""

""" 
random.randint(a,b) geram número inteiros aleatórios entra a e b
random.random() gera um número inteiro aleatório entro 0 e 1
random.choice([a,b,c,d]) escolhe aleatóriamente uma das entras
random.uniform(0,10) geram núemros aléatórios uniformes tbm
""" 

""" Em alguns casos queremos que nosso processo seja reproduzível, 
então queremos que esses números sejam aleatórios, contúdo apresente o mesmo resultado
, assim, podemos colocar uma semente, o exemplo abaixo adiciona uma semente e se colocadas
iguais resultaram no mesmo conjunto de números aleatórios"""


# random.seed(12)
# print([random.uniform(0,10) for x in range(4)])

# random.seed(12)
# print([random.uniform(0,10) for x in range(4)])

""" Esta reproducibilidade é o que representa os números serem pseudoaleatórios """ 


##################### Normal Distribution ########################################################
""" Na função stats.norm.pdf() temos os parâmetros x (cutoff), loc é a média da distribuição 
e scale é o valor do desviopadrão.
"""

prob_under_minus1 = stats.norm.cdf(x = -1,
                                    loc = 0,
                                    scale = 1)
prob_over_1 = 1 - stats.norm.cdf(x = 1,
                              loc = 0,
                              scale = 1)

between_prob = 1 - (prob_over_1 + prob_under_minus1)

print(prob_under_minus1, prob_over_1, between_prob)

# prob_under_minus2 = stats.norm.cdf(x = -2,
#                                     loc = 0,
#                                     scale = 1)
# prob_over_2 = 1 - stats.norm.cdf(x = 2,
#                               loc = 0,
#                               scale = 1)

# betwen_prob = 1 - (prob_over_2 + prob_under_minus2)

# prob_under_minus3 = stats.norm.cdf(x = -3,
#                                    loc = 0,
#                                    scale = 1)
# prob_over_3 = 1 - stats.norm.cdf(x = 3,
#                              loc = 0,
#                              scale = 1)

# betwen_prob = 1 - (prob_over_3 + prob_under_minus3)

"""Os gráficos acima mostram medidas de 1sigma, 2sigma e 3sigma """

""" Podemos ver que aproximadamente 15,9% dos pontos são menor que x = -1 e maior que x = +1, e que a 
probabilidade de encontrar valores entre -1 e 1 é aproximadamente 68% isto é 1sigma"""


# plt.rcParams["figure.figsize"] = (9,9)

# plt.fill_between(x = np.arange(-4,-1, 0.01),
#                  y1 = stats.norm.pdf(np.arange(-4,-1, 0.01)),
#                  facecolor = 'red',
#                  alpha = 0.35)

# plt.fill_between(x = np.arange(1,4,0.01),
#                  y1 = stats.norm.pdf(np.arange(1,4, 0.01)),
#                  facecolor = 'red',
#                  alpha = 0.35)

# plt.fill_between(x = np.arange(-1,1,0.01),
#                  y1 = stats.norm.pdf(np.arange(-1,1,0.01)),
#                  facecolor = 'blue',
#                  alpha = 0.35)

# plt.text(x=-1.8, y=0.03, s= round(prob_under_minus1,3))
# plt.text(x=-0.2, y=0.1, s= round(between_prob,3))
# plt.text(x=1.4, y=0.03, s= round(prob_over_1,3))

print(stats.norm.ppf(q = 0.025))
print(stats.norm.ppf(q = 0.975))


##################### Binomial Distribution ########################################################


""" Binomial Distribution é uma distribuição de probabilidade discreta que modela 
resultados em um dado número de tentativas aleatórias de um evento ou experimento.
A distribuição é definida por dois parâmetros """


