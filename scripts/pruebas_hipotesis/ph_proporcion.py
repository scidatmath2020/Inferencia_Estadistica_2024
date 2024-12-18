# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:19:05 2024

@author: Usuario
"""

from scipy.stats import binomtest

# Datos
x = 45  # Número de éxitos observados
n = 100  # Tamaño de la muestra
p = 0.5  # Proporción esperada bajo H0
alternative = 'two-sided'  # 'two-sided', 'greater' o 'less'

# Prueba binomial
resultado = binomtest(x, n, p, alternative=alternative)

print(f"El valor-p es: {resultado.pvalue}")

#%%

