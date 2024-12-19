# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:53:52 2024

@author: Usuario
"""

'''
###############################################################################
###############################################################################
#################    PRUEBA DE HIPÓTESIS PARA LA VARIANZA    ##################
###############################################################################
###############################################################################
'''

import numpy as np
from scipy.stats import chi2

def chi_squared_test_variance(n, sigma_squared, sigma_squared_0, alternative='two-sided', alpha=0.05):
    """
    Prueba de hipótesis chi-cuadrado para la varianza.
    Supone que la población es normal.

    Parameters:
    - n: tamaño de la muestra.
    - s_squared: varianza muestral.
    - sigma_squared_0: varianza poblacional bajo la hipótesis nula (H0).
    - alternative: tipo de hipótesis ('two-sided', 'greater', 'less').
    - alpha: nivel de significancia (default: 0.05).

    Returns:
    - chi2_stat: valor del estadístico chi-cuadrado.
    - p_value: valor p de la prueba.
    - rejection_region: región de rechazo de H_0.
    - conclusion: decisión sobre H_0 (rechazo o no rechazo).
    """
    # Estadístico chi-cuadrado
    chi2_stat = (n - 1) * sigma_squared / sigma_squared_0

    # Cálculo de la región de rechazo y valor p según la hipótesis alternativa
    if alternative == 'greater':
        chi2_critical = chi2.ppf(1 - alpha, n-1)
        rejection_region = (chi2_critical, np.inf)
        p_value = 1 - chi2.cdf(chi2_stat, n-1)
    elif alternative == 'less':
        chi2_critical = chi2.ppf(alpha, n-1)
        rejection_region = (0, chi2_critical)
        p_value = chi2.cdf(chi2_stat, n-1)
    elif alternative == 'two-sided':
        chi2_critical_low = chi2.ppf(alpha / 2, n-1)
        chi2_critical_high = chi2.ppf(1 - alpha / 2, n-1)
        rejection_region = ((0, chi2_critical_low), (chi2_critical_high, np.inf))
        p_value = 2 * min(chi2.cdf(chi2_stat, n-1), 1 - chi2.cdf(chi2_stat, n-1))
    else:
        raise ValueError("El parámetro 'alternative' debe ser 'two-sided', 'greater' o 'less'.")

    # Decisión basada en el valor p
    if p_value < alpha:
        conclusion = f"Rechazamos H0 al nivel de significancia {alpha}."
    else:
        conclusion = f"No rechazamos H0 al nivel de significancia {alpha}."

    return chi2_stat, p_value, rejection_region, conclusion

#%%

sample = [2.32, 4.26, 4.02, 4.44, 3.68, 2.72, 1.90, 1.21] 
n = len(sample)
s_squared = np.var(sample,ddof=1)
sigma_squared_0 = 0.8**2 
alpha = 0.05
alternative='two-sided'

chi_squared_test_variance(n, s_squared, sigma_squared_0, alternative, alpha)