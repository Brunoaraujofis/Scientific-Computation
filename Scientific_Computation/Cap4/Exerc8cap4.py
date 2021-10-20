# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:13:22 2021

@author: Bruno
"""

import numpy as np
import scipy.linalg as sl

ordem = int(input("Insira a ordem da sua matriz:  "))

A = np.zeros((ordem,ordem))
for i in np.arange(ordem):
    for j in np.arange(ordem):
     if i < j and j != ordem-1:
         A[i,j] = 0
     elif i == j or j == ordem-1:
         A[i,j] = 1
     else: A[i,j] = -1
print(A) 
print('A matriz de escalonamento P é:\n' ,sl.lu(A)[0])
print('A matriz triangular inferior L:\n' ,sl.lu(A)[1])
print('A matriz triagular superior U:\n' ,sl.lu(A)[2])


growthfactor = np.max(np.abs(sl.lu(A)[2]))/np.max(np.abs(A))
print(growthfactor)   

