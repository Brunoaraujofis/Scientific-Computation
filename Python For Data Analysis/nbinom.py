# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:09:31 2021

@author: Bruno
"""

"""Estudando a distribuição binomia negativa"""

from scipy.stats import nbinom
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1,1)

"Definindo os parâmetros da pmf"

n, p = 5, 0.5  

""" n é o número de sucessos e p a probabilidade de sucessos, um exemplo seria 
definir que tirar cara em um lançamento de moeda é sucesso e 
a probabilidade de sucesso é 0.5"""

""""Calculamos os quatro primeiros momentos: média, variancia, 
Skewness ou assimetria e kurtose"""

# mean, var, skew, kurt = nbinom.stats(n,p, moments = "mvsk")

# x = np.arange(nbinom.ppf(0.01,n,p), nbinom.ppf(0.99,n,p))
# ax.plot(x, nbinom.pmf(x,n,p), 'bo', ms = 8, label = 'nbinom pmf') # plot das bolinhas em (x,pmf(x))
# ax.vlines(x,0,nbinom.pmf(x,n,p), color = 'b', lw = 5, alpha = 0.5) # plot das linhas verticais localizadas em x, começando no zero e indo até o pmf(x)

""" De forma alternativa podemos chamar como uma função
 a distribuição(objeto)  isto define um objeto RV frozen"""
 
# rv = nbinom(n,p)
# x = np.arange(nbinom.ppf(0.01,n,p), nbinom.ppf(0.99,n,p))
# ax.vlines(x, 0, rv.pmf(x), colors='k', linestyle='-', lw=1, label='frozen pmf')
# ax.plot(x, nbinom.pmf(x,n,p), 'bo', ms = 8, label = 'nbinom pmf') # plot das bolinhas em (x,pmf(x))
# ax.legend(loc = 'best', frameon = False)
# plt.show()
 

"""Checagem da acuracia de cdf(Cumulative Distribution Function) e ppf(Percent-Point Function)"""

# x = np.arange(nbinom.ppf(0.01,n,p), nbinom.ppf(0.99,n,p))
# prob = nbinom.cdf(x,n,p)
# print(np.allclose(n,nbinom.ppf(prob,n,p)))

""" Gerando números aleatórios"""

# r = nbinom.rvs(n,p, size = 1000)

#####################################################################################
# Methods:

# rvs(n, p, loc=0, size=1, random_state=None) | Random variates.

# pmf(k, n, p, loc=0)| Probability mass function.

# logpmf(k, n, p, loc=0) |Log of the probability mass function.

# cdf(k, n, p, loc=0)| Cumulative distribution function.

# logcdf(k, n, p, loc=0) |Log of the cumulative distribution function.

# sf(k, n, p, loc=0)  | Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).

# logsf(k, n, p, loc=0)| Log of the survival function.

# ppf(q, n, p, loc=0)| Percent point function (inverse of cdf — percentiles).

# isf(q, n, p, loc=0) | Inverse survival function (inverse of sf).

# stats(n, p, loc=0, moments=’mv’)|  Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).

# entropy(n, p, loc=0) |  (Differential) entropy of the RV.

# expect(func, args=(n, p), loc=0, lb=None, ub=None, conditional=False) | Expected value of a function (of one argument) with respect to the distribution.

# median(n, p, loc=0) |  Median of the distribution.

# mean(n, p, loc=0) | Mean of the distribution.

# var(n, p, loc=0) | Variance of the distribution.

# std(n, p, loc=0) |  Standard deviation of the distribution.

# interval(alpha, n, p, loc=0) | Endpoints of the range that contains fraction alpha [0, 1] of the distribution



