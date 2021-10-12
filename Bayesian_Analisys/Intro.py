# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:24:55 2021

@author: Bruno
"""
import scipy as sc
import pymc3 as pm

# mu = 0.
# sigma = 1.
# X = sc.stats.norm(mu, sigma)
# x = X.rvs(3)

# ------------------------------------ Thinking Probabilistically - Cap 1 -------------------------------------------

# Este capítulo trata dos conceitos centrais da Estatística Baysiana, apesar do estarmos 
#interessados também na parte de como fazer estatística Baysiana com python este capítulo é 
# um pouco mais teórico.  

# Queremos Cobrir os seguintes tópicos:
    # Modelagem Estatística
    # Probabilidade e Incerteza
    # Teorama de Bayes' e Inferência Estatística
    # InferÊncia de parâmetro único e o clássico problema da rolagem de moedas
    # Escolhendo piores e porque pessoas não gostam disto
    # Comunicando uma Analise Bayesiana

# ------------------------------------ Statistics, models, and this book's approach -------------------------------------------
    
# Estatistica é sobre coletar, organizar, analisar e interpretar dados e portanto o conhecimento
# estatístico é essencial na analise de dados.

