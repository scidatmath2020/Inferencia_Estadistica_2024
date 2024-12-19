# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:06:53 2024

@author: Usuario
"""

'''
###############################################################################
###############################################################################
################    PRUEBA DE HIPÓTESIS PARA LA PROPORCIÓN    #################
###############################################################################
###############################################################################
'''

from scipy.stats import binom

def exact_test_proportion(n, x, p_0, alternative='two-sided', alpha=0.05):
    """
    Prueba exacta para proporciones basada en la distribución binomial.

    Parameters:
    - n: tamaño de la muestra.
    - x: número de éxitos observados.
    - p_0: proporción poblacional bajo la hipótesis nula (H0).
    - alternative: tipo de hipótesis ('two-sided', 'greater', 'less').
    - alpha: nivel de significancia (default: 0.05).

    Returns:
    - test_stat: valor del estadístico de prueba (número de éxitos observados).
    - p_value: valor p exacto de la prueba.
    - rejection_region: región de rechazo de H_0.
    - conclusion: decisión sobre H_0 (rechazo o no rechazo).
    """
    # El estadístico de prueba es simplemente el número de éxitos observados
    test_stat = x

    # Cálculo del valor p según la hipótesis alternativa
    if alternative == 'greater':
        p_value = 1 - binom.cdf(x - 1, n, p_0)
        rejection_region = f"x >= {binom.ppf(1 - alpha, n, p_0):.0f}"
    elif alternative == 'less':
        p_value = binom.cdf(x, n, p_0)
        rejection_region = f"x <= {binom.ppf(alpha, n, p_0):.0f}"
    elif alternative == 'two-sided':
        # Calculamos ambos extremos para hipótesis bilateral
        p_low = binom.cdf(x, n, p_0)
        p_high = 1 - binom.cdf(x - 1, n, p_0)
        p_value = 2 * min(p_low, p_high)
        rejection_region = f"En ambos extremos, según p-value <= {alpha}"
    else:
        raise ValueError("El parámetro 'alternative' debe ser 'two-sided', 'greater' o 'less'.")

    # Decisión basada en el valor p
    if p_value < alpha:
        conclusion = f"Rechazamos H0 al nivel de significancia {alpha}."
    else:
        conclusion = f"No rechazamos H0 al nivel de significancia {alpha}."

    return test_stat, p_value, rejection_region, conclusion

#%%
n = 100
x = 15
p_0 = 0.2
alternative = "two-sided"
alpha = 0.05 
exact_test_proportion(n, x, p_0, alternative, alpha)
