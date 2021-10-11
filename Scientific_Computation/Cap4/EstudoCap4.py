# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:44:23 2021

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
from scipy.linalg import norm
from scipy.linalg import solve 

# --------------------------- Capítulo 3: Container Types ---------------------------


# Os objetos da algebra linear são os vetores e matrizes.
# O pacote numpy oferece todo o feramental necessário para manejar esses objetos.

# --------------------------- Overview of the array type  ---------------------------

# --------------------------- Vector and Matrices  ---------------------------

v = np.array([1.,2.,3.])

# o objeto v é um vetor e se comporta muito parecido com os vetores da algebra linear.

# Aqui segue algumas operações com vetores:

v1 = np.array([1.,2.,3.])
v2 = np.array([2.,0.,1.])

# Multiplicação por escalares:
print(2*v1)
print(v1/2)

# Combinações lineares:
print(3*v1 + 2*v2)

# Norma de um vetor

print(norm(v1))

# Produto interno

print(np.dot(v1,v2))

# Existem algumas operações que são de elementos por elementos
print(v1*v2)
print(v2/v1)
print(v1 -v2)
print(v1 + v2)
# Algumas funções atuam sobre os elementos de uma array tbm

print(np.cos(v1))

# Um  exemplo de matriz é dado por  

M = np.array([[1.,2.,3.], [4.,3.,2.]])

# Um vetor n dim, uma matriz 1xn ou nx1 são três objetos diferentes. 

# De um vetor v podemos construir matrizes colunas e matrizes linhas

R = np.array([1.,2.,3.]).reshape(1,3)
print(np.shape(R)) # Este comando printa o número de linhas e número de colunas

C = np.array([1.,2.,3.]).reshape(3,1)
print(np.shape(C))

print(np.shape(v))

# --------------------------- Indexing and slices ---------------------------

# Indexando e Slice (recortes) são bem parecidos com os de list

v = np.array([1.,2.,3.])
M = np.array([[3.,4.,10.],[5.,6.,11.]])

print(v[1:])
print(M[0, 0])
print(M[0, 1])
print( M[1]) # Neste caso retorna o vetor v1 = np.array([5.,6.])
print(M[1:]) # Neste caso retorna uma matriz

# --------------------------- Linear algebra operations ---------------------------

# A operação mais comum é dot(M,v) ou alternativamente M@v

print(np.dot(M,v))
print(np.shape(np.dot(M,v)))

# Neste exemplo acima é a atuação da matriz M no vetor v, porém temos o produto interno
# e o produto de matrizes feita da mesma forma, apenas precisa seguir a regra da dimensão
# da multiplicação de matrizes

# ------------------------------------- Solving a Liner System -----------------------------------------------

#Para solucionar equações lineares com matrizes precisamos importar solve do scipy.linalg

# # Exemplo 
# x_1 + 2x_2 = 1
# 3x_1 + 4x_2 = 4,
A = np.array([[1.,2.], [3.,4.]])
b = np.array([1., 4])
x = solve(A,b)
print(x)
print(np.allclose(np.dot(A,x),b))

# O comando allclose compara dois vetores e retorna true caso esses estejam próximos 
# o suficiente, podendo definir um limite de tolerancia

# -------------------------------------The dot operations -----------------------------------------------

# Em python a operação de redução de tensores é feita apartir a função dot

angle = np.pi/3
M = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
v = np.array([1.,0.])
y = np.dot(M,v)
print(y)

# -------------------------------------The array type -----------------------------------------------

# Array são caracterizados basicamente por três propriedades

# shape: Descreve como os dados devem ser interpretados,como vetores, como matrizes,
# ou como tensores de ordem maior. Também fornece a dimensão do objeto
 
# dtype: Fornece o tipo do dado interno ao array ( float, complexo, inteiros e assim em diante)

# strides: Especifica qual a ordem que os dados do array devem ser lidos

# Considere o seguinte array

A = np.array([[1,2,3], [3,4,6]])
print(A.shape) 
print(A.dtype)
print(A.strides)

# Neste caso strides nos diz que para ir para a próxima coluna ele precisa nadar 4bytes 
# na memória equivalentemente 32 bits, já para ir para a próxima linha ele
# precisa andar 32 x 3 = 96 bits ou 12 bytes.

# ------------------------------------- Creating arrays from lists -----------------------------------------------

# A syntax para se criar vetores reais e complexos são dadas a seguir

v1 = np.array([1.,2.,35.], dtype=(float))

v2 = np.array([1., 10. + 1j*25., 1j*2.], dtype=(complex))

print(v1)
print(v2)

# Quando nenhum dtype é especificado o dtype então é adivinhado.
# A função array escolhe o type que pode armazenar a informação em todas as entradas

V = np.array([1., 2])
print(V.dtype)
#No exemplo acima temos um inteiro e um float então o array vai interpretar que a entrada é float

V = np.array([1. + 0j, 1])
print(V.dtype)
# ------------------------------------- Array and Python parentheses -----------------------------------------------

# Python permite quebra de linhas quando temos braces ou parenteses abertos
# Assim uma identação boa para matrizes pode ser usada. e.g.,

A = np.array([[1.,0.], [0.,1.]]) #------->

A = np.array([[1.,0.],
             [0.,1]])
print(A)

# ------------------------------------- Accessing array entries -----------------------------------------------
# As entradas dos arrays são acessadas com index
print(A[0, 0]) # primeira linha, primeira coluna
print(A[-1, 0]) # última linha, primeira coluna

# ------------------------------------- Basic array slicing -----------------------------------------------
M = np.array([[0,1,2,3],
              [4,5,6,7],
              [8,9,10,11]])
# M[i, :]  corre sobre todas as culunas fixado a linha i
print(M[1, :])
print(M[1, :].shape)

# M[i, :] é um vetor de dimesão (4,)

# M[:, j] é um vertor de dimensão (3,)
print(M[:, 2].shape)
# M[:, j] corre sobre as linhas fixado uma coluna

print(M[:, 2])

# M[i:j, :] corre sobre as linhas i, i+1, ..., j-1 e sobre todas as colunas, forncendo uma matriz

print(M[0:2, :])
print(M[ 0:2, :].shape) # é uma matriz (2, 4)

# M[i:j, k:l] correra sobre as linhas i, i+1, ..., j-1 e as colunas
# k, k+1, ..., l-1, fornecendo uma matriz com shape (j -i, l-k), quando usamos i >j, l>k


print(M[0:2, 0:3])
print(M[0:2, 0:3].shape)

print(M[2]) # retorna a segunda lina da matriz M
print(M[1:3]) # retorna umas matriz com as lihas 1 e 2 
# Quando n informado as colunas o Numpy interpreta que deve correr sobre todas colunas

# As regras gerais para slicing em array são dadas a seguir

# Acess| ndim| Kind
# index, index| 0| Scalar
# index, slicing| 1 | Vector
# slicing, index| 1| Vector
# slicing, slicing| 2| Matrix 

print(M[1:2, 1:2])
print(M[1:2, 1:2].shape)
print(M[1,1])
print(M[1,1].shape)

#M[1:2, 1:2] e M[1,1] são objetos diferentes, no primeiro caso este é interpretado por
# uma matriz 1x1 de shape(1,1), já o segundo caso é uma escalar

K = M[1:2, 1:2]
print(K[0, 0])

L = M[1,1]
# print(L[0, 0]) retorna um erro pois um scalar n tem ndim

M = np.array([[0.,1.,2.], 
              [3.,4.,5.], 
              [6.,7.,8.], 
              [9.,10.,11.], 
              [12.,13.,14]])

print(M)

# Podemos alterar elementos em uma matriz, e.g., em uma única entrada

M[1,2] = 2
print(M)

# Tambeém podemos alterar um vetor completo

M[1:2, :] = np.array([1., 2.,3])
print(M)

# Ou até mesmo uma parte completa da matriz

M[1:4, 1:2] = np.array([[0.],
               [0.],
               [0.]])

print(M)

M[1:3, :] = np.array([[0., 0., 0.],
              [0., 0., 0.]])
print(M)

# Note que se tentarmos colocar um vetor como M[1:4, 1:2] retorna um erro,
# isso expressa a diferenã entre uma vetor e uma matriz coluna

# M[1:4, 1:2] = np.array([0., 0., 0.]) -----> retorna um erro pois o vetor é shape (3,)
# e o elemento M[1:4, 1:2] é shape (3,1)

# ------------------------------------- Functions to Construct Arrays -----------------------------------------------

# Existem alguns métodos para gerar arrays específicos como mostrado na tabela a seguir

# | Methods        | Shape   |       Generates         |
# | zeros((n,m))   | (n,m)   | Matrix filled with zeros|
# | ones((n,m))    | (n,m)   | Matrix filled with ones|
# | diag(v,k)      | (n,n)   |(Sub-, super-) diagonal matrix from a vector v|
# | randon.rad(n,m)| (n,m)   | Matrix filled with uniformly distributed random numbers (0,1)|
# | arange(n)      | (n,)    | vector with first n integers|
# | linspace(a,b,n)| (n,)    | Vector with n equispaced points between a and b
v = np.array([1.,1.,1.,1])
print(np.diag(v,0))

# O comando np.diag(v,k) está colocando na coluna k o vetor v em diagonal
# gerando assim uma matriz de shape = (v +k, v +k)

M2 = np.random.rand(5,3)
print(M2)

v1 = np.linspace(0., 45.,8)
print(v1)

# Podemos usar o comando dtype nos comando zeros, ones, arange

M3 = np.zeros((10,10), dtype= float)
print(M3)

# Podemos usar por exemplo para criar uma matriz cheia de zeros de uma dimensão de 
# qualquer outra matriz, exemplo

A = np.array([[1.,2.,3.,4.], 
             [1.,10.,20.,33.],
             [1.,2.5, 27.3, 19.1],
             [0.5, 0.3, 0.2, 12.52]])

Zeros_A= np.zeros(np.shape(A))
print(Zeros_A)

# O comando itentity(n) cria uma matriz identidade (n,n)

I = np.identity(10)
print(I)
print(np.shape(I))