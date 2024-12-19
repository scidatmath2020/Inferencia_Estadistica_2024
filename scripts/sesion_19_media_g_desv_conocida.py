# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:41:49 2024

@author: Usuario
"""

'''
###############################################################################
###############################################################################
########### PRUEBA DE HIPÓTESIS PARA LA MEDIA CON VARIANZA CONOCIDA ###########
##########              POBLACION NORMAL O GRANDE                    ##########
###############################################################################
###############################################################################
'''

import numpy as np
from scipy.stats import norm

def z_test_mean(n, sample_mean, mu_0, sigma, alternative, alpha):
    """
    Prueba de hipótesis Z para la media poblacional con desviación estándar conocida.
    Supone que la población es normal o el tamaño de la muestra es suficientemente grande.

    Parameters:
    - n: tamaño de la muestra.
    - sample_mean: media muestral.
    - mu_0: media poblacional bajo la hipótesis nula (H0).
    - sigma: desviación estándar poblacional conocida.
    - alternative: tipo de hipótesis ('two-sided', 'greater', 'less').
    - alpha: nivel de significancia (default: 0.05).

    Returns:
    - z_stat: valor del estadístico Z.
    - p_value: valor p de la prueba.
    - rejection_region: región de rechazo de H_0.
    - conclusion: decisión sobre H_0 (rechazo o no rechazo).
    """
    # Cálculo del estadístico Z
    z_stat = (sample_mean - mu_0) / (sigma / np.sqrt(n))

    # Cálculo de la región de rechazo y valor p según la hipótesis alternativa
    if alternative == 'greater':
        z_critical = norm.ppf(1 - alpha)
        rejection_region = (z_critical, np.inf)
        p_value = 1 - norm.cdf(z_stat)
    elif alternative == 'less':
        z_critical = norm.ppf(alpha)
        rejection_region = (-np.inf, z_critical)
        p_value = norm.cdf(z_stat)
    elif alternative == 'two-sided':
        z_critical_low = norm.ppf(alpha / 2)
        z_critical_high = norm.ppf(1 - alpha / 2)
        rejection_region = ((-np.inf, z_critical_low), (z_critical_high, np.inf))
        p_value = 2 * min(norm.cdf(z_stat), 1 - norm.cdf(z_stat))
    else:
        raise ValueError("El parámetro 'alternative' debe ser 'two-sided', 'greater' o 'less'.")

    # Decisión basada en el valor p
    if p_value < alpha:
        conclusion = f"Rechazamos H0 al nivel de significancia {alpha}."
    else:
        conclusion = f"No rechazamos H0 al nivel de significancia {alpha}."

    return z_stat, p_value, rejection_region, conclusion

#%%

n = 40
sample_mean = 725
sigma = 102
mu_0 = 670
alpha = 0.01
alternative='greater'

z_test_mean(n, sample_mean, mu_0, sigma, alternative, alpha)

#%%

# z_test_mean(n, sample_mean, mu_0, sigma, alternative='two-sided', alpha=0.05)

muestra = [2150, 1950, 2170, 1860, 2050, 2120, 1920, 1850, 2230]
n = len(muestra)
sample_mean = np.mean(muestra)
mu_0 = 2000
sigma = 210
alternative = "greater"
alpha = 0.05

est_cont, p_val, reg_rech, conclusion = z_test_mean(n, sample_mean, mu_0, sigma, alternative, alpha)





## Ejemplo 1

sample = [2150, 1950, 2170, 1860, 2050, 2120, 1920, 1850, 2230]
n = len(sample)
sample_mean = np.mean(sample)
sigma = 210
mu_0 = 2000
alpha = 0.05
alternative='greater'

est_cont, p_val, reg_rech, conclusion = z_test_mean(n, sample_mean, mu_0, sigma, alternative, alpha)

print(f"El estadístico de contraste es {est_cont}")
print(f"El p-valor es {p_val}")
print(f"La región de rechazo es {reg_rech}")
print(f"Concluímos que {conclusion}")

#%%



