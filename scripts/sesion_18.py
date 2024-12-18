# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:45:49 2024

@author: Usuario
"""
import pandas as pd
import numpy as np
from scipy.stats import t, binom


#%%
### raw string
tabla = pd.read_csv(r"C:\Users\Usuario\Documents\scidata\24_inf_est\proyectos\mtcars.csv")

tabla.head()


#%%

# Parámetros
media_muestral = tabla.mpg.mean()  # media muestral
s = np.sqrt(np.var(tabla.mpg,ddof=1))               # desviación estándar muestral (desconocida)
n = tabla.shape[0]               # tamaño de la muestra
alpha = 0.05         # nivel de significancia para el 95% de confianza

#%%

# Cálculo del valor crítico t (con n-1 grados de libertad)
t_critico = t.ppf(1 - alpha / 2, df=n - 1)

# Cálculo del margen de error
margen_error = t_critico * (s / np.sqrt(n))

# Cálculo del intervalo de confianza
IC_inferior = media_muestral - margen_error
IC_superior = media_muestral + margen_error

# Mostrar los resultados
print(f"Intervalo de confianza para la media: ({IC_inferior:.2f}, {IC_superior:.2f})")

IC_inferior
IC_superior

#%%

# Parámetros
n = 100  # Tamaño de la muestra
x = 30  # Número de éxitos
alpha = 0.05  # Nivel de significancia para el intervalo de confianza del 95%

# Cálculo del intervalo de confianza exacto utilizando la distribución binomial inversa
p0_lower = binom.ppf(alpha / 2, n, x / n) / n
p1_upper = binom.ppf(1 - alpha / 2, n, x / n) / n

# Mostrar los límites del intervalo
print(f"Intervalo de confianza exacto: ({p0_lower:.4f}, {p1_upper:.4f})")

#%%

delincuencia = pd.read_csv(r"C:\Users\Usuario\Documents\scidata\24_inf_est\proyectos\envipe_2022.csv")

delincuencia.head()

delincuencia.shape[0]  #número de filas

delincuencia.FAC_ELE.sum()

delincuencia.victima.sum()


(delincuencia.FAC_ELE*delincuencia.victima).sum()

p = (delincuencia.FAC_ELE*delincuencia.victima).sum()/delincuencia.FAC_ELE.sum()


#%%

envipe_simplificada = delincuencia.victima

n = envipe_simplificada.shape[0]
x = envipe_simplificada.sum()
alpha = 0.01

p0_lower = binom.ppf(alpha / 2, n, x / n) / n
p1_upper = binom.ppf(1 - alpha / 2, n, x / n) / n

# Mostrar los límites del intervalo
print(f"Intervalo de confianza exacto: ({p0_lower:.4f}, {p1_upper:.4f})")

#%%

















