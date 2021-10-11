# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:18:23 2021

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


# --------------------------- Capítulo 3: Container Types ---------------------------

# Usado para agrupar objetos. A principal diferença entre Container Types esta 
# na forma que os elementos individuais podem ser acessados e nas operações que
# podem ser definidas. 



#-----------------------------------------  LISTAS ------------------------------------------------------

# Uma lista é propriamente como o nome diz, definimos a seguir duas listas;

L = ['a', 20.0, 5]
M = [3, ['a', -3.0, 5]]

# Os elementos individuais são idexados por 0,1,...
# L[i] , i = 0,1,2,... chama o i-ésimmo elemento da lista e.g. L[0] = 'a', 
# M[1] = ['a',-3.0, 5]

print(L[0])
print(M[1])
print(L[2]) 

# A lista M possui uma lista dentro dela, assim com dois indices podemos acessar
# os elementos dentro da lista interior à M, e.g., M[1][0] = 'a'.

print(M[1][0])
print(M[1][1])

# Uma lista com subsequente números inteiros pode ser gerada facilmente pelo
# comando list(range):

L1 = list(range(10))
print(L1)

# Outra forma de gerar lista pode ser dando o começo, fim e o passo da lista, e.g.,
# L2 = list(range(10,30,2))

L2 = list(range(10,32,2))
print(L2)


# Podemos ver o tamanho (número de elementos) de uma lista pelo comando len:

print(len(L2))
print(len(L1))

# ------------------------------------------ SLICING ----------------------------------------------------------------

# Slicing ou repartir uma lista entre os elementos i e j crea uma nova lista 
# onde o primeiro elemento da nova lista é o i-ésimo elemento da lista antiga 
# e o último elemento é o j-ésimo elemento da lista antiga. 

# o comando L[i:j] cria uma nova lista pegando do i-ésimo elemento até o (j-1)-ésimo elemento


l = list(range(10))
print(l)
print(l[2:7])
print(l[:7])
print(l[2:-1])
print(l[:-2])

# l[i:] significa remova os i-ésimos primeiros elementos, l[:i] significa pegue
# os i-ésimos elementos, já l[:-i] significa remova os i-ésimos elementos finais
# e l[-i:] pegue os i-ésimos elementos finais. Podemos combinar os comando por 
# l[i:] and l[:-j] significando remova os i-ésimos elementos iniciais e também 
# remova os j-ésimos elementos finais. 

print(l[:-7])
print(l[:7])
print(l[-7:])
print(l[3:-2])

#Estes são exemplos de como usar Slicing. 

Lc = ['C','l','o','u','d','s']
print(Lc[1:5])
print(Lc[1:])
print(Lc[:])
print(Lc[1:])
a = [1,2,3]
for iteration in range(4):
        print(np.sum(a[0: iteration-1]))

# ------------------------------------------ Strides ----------------------------------------------------------------

# Strides ou passos servem para especificar os passos quando estamos computando slices (fatias). 
# Segue alguns exemplos de Strides usado em Slices.

L = list(range(100))
print(L)
print(L[:10:2])
print(L[::20])
print(L[10:20:3])

# Note que podemos usar Strides negativos

print(L[20:10:-3])

L = [1,2,3]
R = L[::-1]
print(R)

# ------------------------------------------ Altering Lists ----------------------------------------------------------------

# Algumas operações padrões com lista são incerção e remoção de elementos, além 
# da concatenação de listas. Com o uso do slicing incerção de elementos e exclusão de
# elementos se torna direta, para exclusão basta substituir por uma lista vazia.

L = ['a', 1, 2, 3, 4]
L[2:3] = []
print(L)

L = list(range(30))
print(L)
L[0:10] = []
print(L)

L = ['a', 1, 2, 3]
L[1:2] = []
print(L)
L[1:1]= [1,4]
print(L)
print(len(L))

# A soma de listas é chamada de concatenação

L = [1, -17]
M = [23.5, 18.3, 5.0]

print(L + M)
print( M + L)

# De modo geral a soma de lista não é comutativa

# A multiplicação de lista é a concatenação de uma lista com ela mesma n vezes

n = 3
L = ['a',1 ,2,3]

print(n*L)
print(L*n)

m = 20
print( m*[0])

# ------------------------------------------ Belonging to a List ----------------------------------------------------------------

# Belonging to a list ou pertencente à uma lista, pode usar as keywords in e not in 
# para ver se um elemento pertence a alguma lista.

L = list(range(4))
print(L)
print( 0 in L)
print(4 in L)
print(4 not in L)

# ------------------------------------------ List Methods ----------------------------------------------------------------

# Existem alguns methods aplicados a lista como os a seguir 

# L.append(x) adiciona o elemento x no fim da lista

L = list(range(4))
L.append("a")
print(L)

# L.extend(M) insere uma nova lista no fim da lista L sendo equivalente a soma

L = list(range(4))
M = list(range(10))
M = M[4:10]
print(M)
L.extend(M)
print(L)

# L.insert(i,x) insere o elemento x na i-ésima entrada

L = list(range(10))
print(L) 
L.insert(1, "A")
print(L)

# L.remove(x) remove o primeiro x que encontrar na lista

L = list(range(6))
L.insert(3, 0)
L.remove(0)
print(L)

# L.count(x) retorna o número de vezes que x aparece em L

L = list(range(5))
print(L.count(5))
print(L.count(4))
L.insert(3,0)
L.insert(3,0)
L.insert(3,0)
L.insert(3,1)
print(L)
print(L.count(0))

# L.sort() ordena ou classifica os elementos de L.

L = [11, 3 , 27, 49, 50,0 , 22, 7, 11, 6, 49, 70]
print(L)
L.sort()
print(L)



# L.reverse reverte os elementos da lista

L.reverse()
print(L)

# L.pop remove o último elemento da lista

L.pop()
print(L)

 # ------------------------------------------ Merging Lists - zip ----------------------------------------------------------------

# Incorporar uma lista ou "Merging list" é útil e a partir de uma lista podemos
# formar uma tupla

ind = list(range(5))
color = ['red','green','blue', 'alpha']
print(color)
print(list(zip(color, ind)))

# O exemplo também ilustra o que acontece se as listas tem tamanhos diferentes
# neste caso, a tupla fica do tamanho da menor lista.

L = ['a', 'b', 'c']
print(list(zip(color, ind, L)))

L1 = list(zip(color, ind, L))
print(L1[1])

# O comando zip crita um objeto operacional (Iterable object) que pode ser transformado em uma
# lista como no caso anterior.

 # ------------------------------------------ List Comprehension ----------------------------------------------------------------

# O list comprehension permite criar uma lista usando condicionais.
# a syntax é dada a seguir: [<expr> for <variable> in <list>]
# ou de forma mais geral [<expr> for <variable> in <list> if <condition>]

L = [2,3,10,1,5]
L2 = [x**2 for x in L]
print(L2)
L3 =  [x**2 for x in L if 4 < x <=10]
print(L3)

# É possível ter vários lopps em List Comprehension

L = list(range(3))
print(L)
L.append(3)
L = L[1:4]
print(L)
L1 = list(range(7))
L1 = L1[4:7]
print(L1)
M = []
M.append(L)
M.append(L1)
print(M)
print(M[1][1])
flat = [M[i][j] for i in range(2) for j in range(3)]
print(flat)

# List Comprehension é estritamente relacionada com a notação de conjuntos matemática
# por exempli L_{2} = {2x : x pertence L} em matemática é o mesmo que 
# L2 = [2*x for x in L]. Uma grande diferença é que lista é ordenada enquanto 
# conjuntos não são necessáriamente.

 # ------------------------------------------ Arrays ----------------------------------------------------------------

# O pacote Numpy oferece arrays, que são container structures para a manipulação
# de vetores, matrizes, ou até mesmo tensores de maiores ordens. 
# Nesta seção vamos entender analogias entre arrays e lists
# Arrays são construídas a partir de listas pela função array

v = np.array([1.,2.,3.])
A = np.array([[1.,2.,3.],[4.,5.,6.]]) 

# Para acessar um elemento de um vetor precisamos dar uma indice
# já para acessar um elemento de uma matriz precisamos dar dois indices

print(v[2])
print(A[1][2])

#Algumas propriedades dos arrays vectors e matrices e alguns casos que 
# estes se diferem de lists são dados a seguir.

# O acesso a elementres de uma array são equivalentes a de lists

M = np.array([[1.,2.],[3.,4.]])
v = np.array([1.,2.,3.])

print(v[0])
print(v[:2])
print(M[0][1])
v[:2] = [10.,20.]
print(v)

# O número de elementos em um vetor ou o número de linhas em uma matriz 
# são obtidos pela função len

print(len(v))
print(len(M))

# Temos as operação +, *,/ e - que são definidas termo a termo

M1 = np.array([[1., 2.],[3.,4]])
M2 = np.array([[-1., -2.],[-3.,-4.]])

M = M1 + M2
print(M)

M = M1 * M2
print(M)

M = M1 / M2
print(M)

M = M1 - M2
print(M)

# Alem da multiplicação por escalar usual e o operador @ é usado para 
# multiplicação de matrizes

M = M1@M2
print(M)

# Não existe append method para arrays


 # ------------------------------------------ Tuples ----------------------------------------------------------------
# Uma tuple é uma lista imutável. 

my_tuple = (0, 1, 2)

print(my_tuple)
# my_tuple[0] = 'a' retorna um erro 


 # ------------------------------------------ Dictionaries ----------------------------------------------------------------


# Listas, tuples e arrays são objetos ordenados. Os objetos individuais (componetes)
# são acessados, processados etc de acordo com a sua localização na lista. 
# Já dictionaries são não ordenados conjuntos. Acessando os dados de um dictionari
# por chaves

truck_whell = {'name': 'whell', 'mass': 5.7, 
'Ix':20.0, 'Iy' : 1.0, 'Iz' : 17.0, 'center of mass' : [0., 0., 0.]}

# uma Key/data é indicado pelo ponto duplo : e os elementos indivíduais 
# são acessados por sua keys

print(truck_whell['name'])
print(truck_whell['Iz'])
print(truck_whell['center of mass'])

# Podemos adicionar novos objetos ao dictionary criando uma nova key

truck_whell['Ixy'] = 10.2

print(truck_whell)


# As keys ultilizadas não pode ser listar. 
# O comando dict cria um dictionary a partir de uma lista de tuplas
L1 = list(truck_whell)
print(L1)
L2 = [truck_whell['name'], truck_whell['mass'], truck_whell['Ix'], truck_whell['Iy'],
      truck_whell['Iz'], truck_whell['center of mass'], truck_whell['Ixy']]

print(len(L1) == len(L2))
L3 = list(zip(L1,L2))
print(L3)
truck_whell = dict(L3)
print(truck_whell)

 # ------------------------------------------Looping Over Dictionaries ----------------------------------------------------------------

# Existem três formas principais de se fazer loops com Dictionaries. 

# Looping com a key>
    
for key in truck_whell.keys():
    print(key)
    

# Looping com os valores>

for value in truck_whell.values():
    print(value)

# Looping com itens (key/value)>

for item in truck_whell.items():
    print(item)    

 # ------------------------------------------Sets ----------------------------------------------------------------

# Sets são objetos "containers" que possuem propriedades e operações parecidas
# com as de sets (conjuntos) matemáticos.

# A seguir temos algumas expressões matemáticas de sets e suas versões em python

# A = {1,2,3,4}, B = {5}, C = A união B, D = A intersecção C,
# E = C\A , 5 pertence a C

A = {1,2,3,4}
print(A)
B = {5}
print(B)
C = A.union(B)
print(C)
D = A.intersection(C)
print(D)
E = C.difference(A)
print(E)
print(5 in C)

# Sets contem um elemento uma única vez por definição

A = {1,2,3,2,3,3}
print(A)
B = {1,2,3}
print(A == B)

# Sets são objetos não ordenados

A = {1,2,3}
B = {1,3,2}
print(A == B)

# Podemos comprar sets usando os methods .issubset e .issuperset

A = {2,4}

print(A.issubset({1,2,3,4,5,6}))

print(A.issuperset({2}))
print({1,2,3,4,5,6}.issuperset(A))

# o conjunto vazio é definido no python por

empty_set = set([])
print(empty_set)

print(A.union(empty_set))


#------------------------------------------Container conversions ----------------------------------------------------------------

# A seguir temos as principais propriedade de containers types

# Type: List, Acess: Index, Order: Yes, Duplicate Values: Yes, Mutability: Yes
# Type: Tuple, Acess: Index, Order: Yes, Duplicate Values: Yes, Mutability: No
# Type: Dict, Acess: Key, Order: No, Duplicate Values: Yes, Mutability: Yes
# Type: Set, Acess: No, Order: No, Duplicate Values: No, Mutability: Yes

# Devido as propriedades diferentes dos containers type nos convertemos rotineiramente
# de um type para outro. A seguir temos qual conversão é permitida 
# e sua respectiva syntax

# List -----> Tuple, Syntax: tuple([1,2,3])
# Tuple -----> List, Syntax: List((1,2,3))
# List, Tuple  -----> Set, Syntax: set([1,2,3]), set((1,2,3))
# Set -----> List, Syntax: List({1,2,3})
# Dict -----> List, Syntax: {'a': 4}.values()
# List -----> Dict, Syntax: --- não é possível. 

#------------------------------------------Type Checking ----------------------------------------------------------------

# Podemos usar o comando type para verificar o tipo de variável que estamos lidando

label = 'local error'
print(type(label))

x = [1,2]
print(type(x))

y = (1,2)
print(type(y))

# Contudo se quiser comparar um tipo de variável usamos isinstance 
k = isinstance(x, list)
m = isinstance(x, tuple)
print(k)
print(m)

