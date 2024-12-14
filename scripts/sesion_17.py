# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:03:21 2024

@author: Usuario
"""

import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy.stats import t

#%%

'''Intervalo de confianza para la media poblacional con varianza conocida (población normal)'''


# Parámetros
media_muestral = 50  # media muestral
sigma = 10           # desviación estándar de la población
n = 30               # tamaño de la muestra
alpha = 0.05         # nivel de significancia para el 95% de confianza

# Cálculo del valor crítico Z
Z = norm.ppf(1 - alpha / 2)

# Cálculo del margen de error
margen_error = Z * (sigma / np.sqrt(n))

# Cálculo del intervalo de confianza
IC_inferior = media_muestral - margen_error
IC_superior = media_muestral + margen_error

# Mostrar los resultados
print(f"Intervalo de confianza para la media: ({IC_inferior:.2f}, {IC_superior:.2f})")


#%%

poblacion = pd.read_csv("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est\\proyectos\\tabla_poblacion_simulada.csv")



# Tomar una muestra aleatoria; random_state es para establecer reproductibilidad
muestra = poblacion.sample(n=n,random_state=2024)

n = len(muestra)
media_muestral = muestra["sd_4"].mean()
sigma = 4
alpha = 0.05

# Cálculo del valor crítico Z
Z = norm.ppf(1 - alpha / 2)

# Cálculo del margen de error
margen_error = Z * (sigma / np.sqrt(n))

# Cálculo del intervalo de confianza
IC_inferior = media_muestral - margen_error
IC_superior = media_muestral + margen_error

# Mostrar los resultados
print(f"Intervalo de confianza para la media: ({IC_inferior:.2f}, {IC_superior:.2f})")

#%%

def selecciones(alpha):
    n = 25
    muestra = poblacion.sample(n=n)
    media_muestral = muestra["sd_4"].mean()
    sigma = 4

    # Cálculo del valor crítico Z
    Z = norm.ppf(1 - alpha / 2)

    # Cálculo del margen de error
    margen_error = Z * (sigma / np.sqrt(n))

    # Cálculo del intervalo de confianza
    IC_inferior = media_muestral - margen_error
    IC_superior = media_muestral + margen_error
    return [IC_inferior,IC_superior]

#%%

CI = [selecciones(0.01) for x in range(1000)]
resultados = [1 if x[0]<20 and 20<x[1] else 0 for x in CI]

intervalos = pd.DataFrame({"int_inf":[x[0] for x in CI],"int_sup":[x[1] for x in CI]})

intervalos["le_atine"] = resultados

intervalos["le_atine"].sum()

#%%

'''Intervalo de confianza para la media poblacional con varianza desconocida (población normal)'''

media_muestral = 50  # media muestral
s = 10               # desviación estándar muestral (desconocida)
n = 30               # tamaño de la muestra
alpha = 0.05         # nivel de significancia para el 95% de confianza

# Cálculo del valor crítico t (con n-1 grados de libertad)
t_critico = t.ppf(1 - alpha / 2, df=n - 1)

# Cálculo del margen de error
margen_error = t_critico * (s / np.sqrt(n))

# Cálculo del intervalo de confianza
IC_inferior = media_muestral - margen_error
IC_superior = media_muestral + margen_error

# Mostrar los resultados
print(f"Intervalo de confianza para la media: ({IC_inferior:.2f}, {IC_superior:.2f})")

#%%

poblacion = pd.read_csv("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est\\proyectos\\tabla_poblacion_simulada.csv")
n = 25

# Tomar una muestra aleatoria; random_state es para establecer reproductibilidad
muestra = poblacion.sample(n=n,random_state=2024)

n = len(muestra)
media_muestral = muestra["sd_desconocido"].mean()
s = np.std(muestra["sd_desconocido"],
           ddof=1)
alpha = 0.05    

# Cálculo del valor crítico t (con n-1 grados de libertad)
t_critico = t.ppf(1 - alpha / 2, df=n - 1)

# Cálculo del margen de error
margen_error = t_critico * (s / np.sqrt(n))

# Cálculo del intervalo de confianza
IC_inferior = media_muestral - margen_error
IC_superior = media_muestral + margen_error

# Mostrar los resultados
print(f"Intervalo de confianza para la media: ({IC_inferior:.2f}, {IC_superior:.2f})")
