# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:54:59 2024

@author: Usuario
"""

'''
###############################################################################
###############################################################################
######### PRUEBA DE HIPÓTESIS PARA LA MEDIA CON VARIANZA DESCONOCIDA ##########
#########                       POBLACION CUALQUIERA                 ##########
###############################################################################
###############################################################################
'''

import numpy as np
from scipy.stats import t

def t_test_mean(n, sample_mean, sample_std, mu_0, alternative, alpha):
    """
    Prueba de hipótesis t para la media poblacional con desviación estándar desconocida.
    Supone que la población es normal o el tamaño de la muestra es suficientemente grande.

    Parameters:
    - n: tamaño de la muestra.
    - sample_mean: media muestral.
    - sample_std: desviación estándar muestral.
    - mu_0: media poblacional bajo la hipótesis nula (H0).
    - alternative: tipo de hipótesis ('two-sided', 'greater', 'less').
    - alpha: nivel de significancia (default: 0.05).

    Returns:
    - t_stat: valor del estadístico t.
    - p_value: valor p de la prueba.
    - rejection_region: región de rechazo de H_0.
    - conclusion: decisión sobre H_0 (rechazo o no rechazo).
    """
    # Grados de libertad
    df = n - 1

    # Cálculo del estadístico t
    t_stat = (sample_mean - mu_0) / (sample_std / np.sqrt(n))

    # Cálculo de la región de rechazo y valor p según la hipótesis alternativa
    if alternative == 'greater':
        t_critical = t.ppf(1 - alpha, df)
        rejection_region = (t_critical, np.inf)
        p_value = 1 - t.cdf(t_stat, df)
    elif alternative == 'less':
        t_critical = t.ppf(alpha, df)
        rejection_region = (-np.inf, t_critical)
        p_value = t.cdf(t_stat, df)
    elif alternative == 'two-sided':
        t_critical_low = t.ppf(alpha / 2, df)
        t_critical_high = t.ppf(1 - alpha / 2, df)
        rejection_region = ((-np.inf, t_critical_low), (t_critical_high, np.inf))
        p_value = 2 * min(t.cdf(t_stat, df), 1 - t.cdf(t_stat, df))
    else:
        raise ValueError("El parámetro 'alternative' debe ser 'two-sided', 'greater' o 'less'.")

    # Decisión basada en el valor p
    if p_value < alpha:
        conclusion = f"Rechazamos H0 al nivel de significancia {alpha}."
    else:
        conclusion = f"No rechazamos H0 al nivel de significancia {alpha}."

    return t_stat, p_value, rejection_region, conclusion

#%%

n = 50
sample_mean = 871
sample_std = 21
mu_0 = 880
alpha = 0.05
alternative = "two-sided"

t_test_mean(n, sample_mean, sample_std, mu_0, alternative, alpha)

#%%

#### Ejemplo 2

muestra = [168, 168, 169, 167, 171, 172, 182, 175, 177, 173, 168, 195, 167, 175, 175]

from plotnine import *
import pandas as pd

datos = pd.DataFrame({"valor":muestra})

ggplot(data = datos) + geom_histogram(mapping=aes(x="valor"),binwidth=4)

#t_test_mean(n, sample_mean, sample_std, mu_0, alternative='two-sided', alpha=0.05)


n = len(muestra)
sample_mean = np.mean(muestra)
sample_std = np.std(muestra,ddof=1)
mu_0 = 180
alternative = "less"
alpha = 0.05

est_cont, p_val, reg_rech, conclusion = t_test_mean(n, sample_mean, sample_std, mu_0, alternative, alpha=0.05)

print(f"El estadístico de contraste es {est_cont}")
print(f"El p-valor es {p_val}")
print(f"La región de rechazo es {reg_rech}")
print(f"Concluímos que {conclusion}")


#%%%

#### Ejemplo 5


muestra = [ 310, 311, 412, 368, 447, 376, 303, 410, 365, 350]

# t_test_mean(n, sample_mean, sample_std, mu_0, alternative, alpha)

n = len(muestra)
sample_mean = np.mean(muestra)
sample_std = np.std(muestra,ddof=1)
mu_0 = 400
alternative = "two-sided"
alpha = 0.05

est_cont, p_val, reg_rech, conclusion = t_test_mean(n, sample_mean, sample_std, mu_0, alternative, alpha=0.05)

print(f"El estadístico de contraste es {est_cont}")
print(f"El p-valor es {p_val}")
print(f"La región de rechazo es {reg_rech}")
print(f"Concluímos que {conclusion}")

#%%

##### Ejemplo 6

n = 50
sample_mean = 871
sample_std = 21
alpha = 0.05
mu_0 = 880
alternative = "two-sided"

est_cont, p_val, reg_rech, conclusion = t_test_mean(n, sample_mean, sample_std, mu_0, alternative,alpha)

print(f"El estadístico de contraste es {est_cont}")
print(f"El p-valor es {p_val}")
print(f"La región de rechazo es {reg_rech}")
print(f"Concluímos que {conclusion}")











#%%

