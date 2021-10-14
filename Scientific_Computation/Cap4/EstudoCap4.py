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
import scipy.linalg as sl
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

# O comando identity(n) cria uma matriz identidade (n,n)

I = np.identity(10)
print(I)
print(np.shape(I))

# ------------------------------------- The shape function -----------------------------------------------

# O shape de uma matriz é uma tupla com suas dimensções. 
M = np.identity(3)
print(np.shape(M))

# Já para um vetor o shape é um singleton (singleton = 1-tupla) contando seu comprimento

v = np.array([1.,0.,0.])
print(np.shape(v))
print(v.shape)
# É equivalente usar o atributo ou a função shape, porém uma das vantagens de usar
# a função shape é que esta pode ser aplicadas em escalares ou listas

print(np.shape(1.))
print(np.shape([1,2]))
print(np.shape([[1,2]]))

# ------------------------------------- Number of Dimensions -----------------------------------------------

# A dimensão de um array pode ser obtida pela função np.ndim()

M = np.identity(7)
print(np.ndim(M))

# Neste caso retorna o núemro dois pois a dim é entendida como o rank do tensor,
# ou ainda o número de indices necessários para obtermos a informação em uma das entradas
# do array

v = np.array([1.,2.,3.])
print(np.ndim(v))
a = 3 # Dimensão de um vetor
print(np.ndim(a)) # Dimensão de um escalar

T = np.zeros((2,3,4))
print(T)
# Neste caso ao printarmos T ele entende com camadas de profundidade de um matriz
# T é um tensor de rank 3, ou seja, um cubo de números.

print(np.ndim(T))

# De forma equivalente como o função shape() retorna uma tupla que tem o número de entradas
# igual a dimensão, então podemos usar len(shape()) para obter a dimensão

print(len(np.shape(T)))
print(len(np.shape(v)))

# ------------------------------------- Reshape -----------------------------------------------


# O método reshape da uma nova vista\formato ao array sem copiar os dados do array, e.g.,
v = np.array([0,1,2,3,4,5])
M = v.reshape(2,3)
print(np.shape(M))
print(M)
print(v)

# Neste caso vimos que podemos construir uma matriz de shape (2,3) de um vetor
M[0,0] = 10
print(v[0])

# Contudo os dados de M e v não são copiados, eles são os mesmos dados
# por isso ao impormos M[0,0] = 0 consequentemente obtemos que v[0] = 0, os dados estão vinculados


# As vezes é conveniente especificar apenas um parâmetro do reshape e deixar o python determinar o outro
# isto é feito colocando o valor de -1 na entrada que vamos deixar em aberto
v = np.arange(8)
print(v)
M = v.reshape(2,-1)
print(np.shape(M))

N = v.reshape(-1,2)
print(np.shape(N))

# -------------------------------------Transpose -----------------------------------------------

# A transposta de uma matriz A pode ser obitida por B = A.T, assim B é a matriz transposta de A
A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])

B = A.T
print(B)
print(A)

# Da mesma forma que o reshape() transposição não copia os dados fazendo com que eles fique vinculados

A[1,0] = 5
# A[1,0] = 5 faz com que B[0,1] =5 
print(B)

# Transpor vetores não fazem sentido já que os vetores são de dimensão 1
# Contudo Numpy ira retornar o mesmo objeto

v = np.arange(4)
print(np.shape(v))
v1 = v.T
print(v1)
print(v)

print(np.shape(v1))

# Podemos criar matrizes coluna ou matrizes linha com vetores

v2 = v.reshape(1,4)
v3 = v.reshape(4,1)
print(v2)
print(v3)

# -------------------------------------Stacking \ Empilhamento -----------------------------------------------

# O método universal de construir matrizes de submatrizes é usar o comando concatene. 
# Com a seguinte sintaxe
a1 = np.array([[1.,2.,3.],
              [0.,0.2,10.]])
a2 = np.identity(3)
a3 = np.concatenate((a1,a2), axis = 0)
print(a3)

# Este comando concatena as matrizes verticalmente quando usamos axis = 0

# Se tentamos o comado a3 = np.concatenate((a1,a2), axis = 1) retorna um erro pois as matrizes não são 
# de tamanhos compatíveis verticalmente

a1= np.array([[1.,2.,3.], [1.,1.,1.], [ 1.,1.,1.]])

a3 = np.concatenate((a1,a2), axis = 1)
print(a3)

# definindo a1 como uma matriz 3x3 conseguimos fazer a concatenação
# Podemos usar as funções np.vstack([,]), np.hstack([,]) e np.columnstack([,])
# para fazer stack vertical, horizontal e de vetores

print(np.vstack([a1,a2]))
print(np.hstack([a1,a2]))

# Assim podemos defirnir novas matrizes e acessalas pelo index da entrada
a4 = np.vstack([a1,a2]) 
print(a4[3,0])

# -------------------------------------Stacking de Vetores -----------------------------------------------

v1 = np.array([1.,2.,3.])
v2 = np.array([0.,0.,0.])
print(np.column_stack([v1,v2]))
print(np.hstack([v1,v2]))
print(np.vstack([v1,v2]))

print(np.shape(np.column_stack([v1,v2]))) # Retorna uma matriz
print(np.shape(np.hstack([v1,v2]))) # Retorna um vetor
print(np.shape(np.vstack([v1,v2]))) # Retorna uma matriz

# Permutação simplética

def symp(v):
    n = len(v)//2
    return np.hstack([v[n:],v[:n]])


v = np.arange(8)
print(v)
print(symp(v))

# -------------------------------------Functions Acting on Arrays -----------------------------------------------

# Existem vários tipos de funções atuando em arrays, algumas atuam por elemento, 
#retornando assim um array do mesmo shape. Estas são chamadas funções universais

# Outras funções de arrays podem retornar arrays de shape diferentes ao original.

# A função cos do numpy é um exemplo clássico de universal function

print(np.cos(np.pi))
a1 = np.array([1, 0, -1]) 
print(np.cos(a1))
print(np.shape(np.cos(np.pi)), np.shape(np.cos(a1)))
print(np.shape(np.pi), np.shape(a1))

# Vemoms que a função cos do numpy não muda o shape do array que ela atua

# -------------------------------------Create universal functions -----------------------------------------------

# Sua função será automaticamente universal se vc usar na construção dela 
# apenas funções universais.

def heaviside(x):
    if x >= 0:
        return 1.
    else:
        return 0.
#print(heaviside(array([-1, 2]))) retorna um erro

# Esperariamos que a função heaviside atuando sobre um array [a,b]
# deveria retorna outro array [heaviside(a),heviside(b)], mas isso não ocorre pois ela 
# sempre retorna uma escalar não importando o tamanho do input que vc coloque nela

# A função do Numpy vectorize nos ajuda a solucionar esse problema rapidamente

vheaviside = np.vectorize(heaviside)

a4= vheaviside(np.array([-1,2]))
print(a4)

# Este método é interessante quando aplicado para plot de funções

xvals = np.linspace(-1, 1, 100)
plt.plot(xvals, np.vectorize(heaviside)(xvals)) # Esta linha de código diz para plotar o xvals pela função vetorizes da heaviside calculada em xvals
plt.axis([-1.5, 1.5, -0.5,1.5])


# -------------------------------------Array functions -----------------------------------------------

# Existem varios exemplos de funções que atuam sobre arrays mas não atuam elemento por elemento
# alguns exemplos são as funções max, min e sum. Essas funções podem atuar em toda a matriz
# ou nas colunas ou linhas. Considerando a convenção de que quando nenhuma 
# argumento é dado atua sobre toda a matriz

A = np.array([[1.,2.,3.,4. ],
             [5.,6.,7.,8.]])
print(np.sum(A))
# O comando possui um parâmetro opicional, axis. Permitindo escolher em qual eixo vamos performar a soma
# E.g., se axis = 0 isto significa que a soma deve ocorrer sobre o primeiro axis ou seja 
# as linhas se soman, no caso axis = 1 as colunas se soman

print(np.sum(A, axis = 0))
print(np.sum(A, axis = 1))

print(np.shape(np.sum(A, axis = 0)))
print(np.shape(np.sum(A, axis = 1)))

# Vale notar que np.sum(B, axis = 0) em que B tem shape = (m,n) retorna um vetor de shape = (n,)
# Em análogo np.sum(B, axis = 1) retorna um vetor de shape = (m,)

# -------------------------------------Linear algebra methods in SciPy -----------------------------------------------
# o módulo scipy.linalg tem um amontoado de métodos para algebra linear em python


# Considere o sistema de equação linear Ax= b, A uma matriz nxn 
#Decomposição LU (lower Upper) decompõe a matriz A em um produto de duas matriz 
# L e U , A = LU onde L é uma matriz triangular inferior e U é uma matriz triangular superior
# Podemos levar enconta tbm que se P é uma matriz de permutação

# Ax = b tem a mesma solução que PAx = Pb
# A decomposição LU encontra uma matriz de permutação P e as matriz L e U de modo que 

# PA = LU ou equivalentemente A = PLU. Além disso, L pode ser determinado de modo que 
# L_{ii} = 1 e como L_{ij} = 0 se i > j temos que determinar apenas L_{ij} para i<j. Assim,
# U tem n + n(n-1)/2 informações e L apenas n(n-1)/2 e portanto L e U pondem ser armazenadas 
# em uma matriz nxn ( Por isso A é nxn). Já a matriz P é alocada em um vetor chado pivot de n entradas

# Scipy tem dois métodos de caucular a fatorização LU. O padrão é scipy.linalg.lu
#Na qual retorna três matrizes P, L e U

# M = np.array([[4.,3.],
#               [6.,3.]])
# print(sl.lu(A))


# o outro método é o lu_factor que é conveniente usar quando usaramos tbm o lu_solve que iremos
# usar a seguir
M = np.array([[4.,3.],
              [6.,3.]])
[LU,piv] = sl.lu_factor(M)
print([LU,piv])

#Aqui, a matriz A é fatorizada e um array com as informações sobre L e U são retornadas, além disso
# é retornado tbm a informação do vetor pivot. Com essas informações o sistema pode ser resolvid
# fazendo uma permutação de linhas nos vetores b de acordo com a informação no vetor pivot.
# Este procedimento é realizado no Python com lu_solve method. 

# Usamos sl.solve((LU,piv),bi) para solucionar o sistema de equações Axi = bi
# onde já performações fatorização LU em A como mostramos no exemplo acima com a matriz M.

# -------------------------------------Solving a least square problem with SVD -----------------------------------------------
