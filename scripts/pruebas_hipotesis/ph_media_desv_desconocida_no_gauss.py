# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:48:52 2024

@author: Usuario
"""

'''
###############################################################################
###############################################################################
######### PRUEBA DE HIPÓTESIS PARA LA MEDIA CON VARIANZA DESCONOCIDA ##########
#########                       POBLACION GENERAL                    ##########
#########                        MUESTRA GRANDE                      ##########
###############################################################################
###############################################################################
'''

from scipy.stats import t
import numpy as np

def t_test_large_sample(sample, mu0, alternative='two-sided', alpha=0.05):
    """
    Prueba de hipótesis t para la media con desviación estándar desconocida (muestra grande).
    Supone que la población no es necesariamente normal.

    Parameters:
    - sample: lista o arreglo con los datos muestrales.
    - mu0: media poblacional bajo la hipótesis nula (H0).
    - alternative: tipo de hipótesis ('two-sided', 'greater', 'less').
    - alpha: nivel de significancia (default: 0.05).

    Returns:
    - t_stat: valor del estadístico t.
    - p_value: valor p de la prueba.
    - conclusion: decisión basada en el nivel de significancia.
    """
    n = len(sample)                        # Tamaño de la muestra
    x_bar = np.mean(sample)                # Media muestral
    s = np.std(sample, ddof=1)             # Desviación estándar muestral
    t_stat = (x_bar - mu0) / (s / np.sqrt(n))  # Estadístico t
    df = n - 1                             # Grados de libertad

    # Cálculo del valor p según la hipótesis alternativa
    if alternative == 'two-sided':
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        p_value = 1 - t.cdf(t_stat, df)
    elif alternative == 'less':
        p_value = t.cdf(t_stat, df)
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'")

    # Decisión basada en el nivel de significancia
    if p_value < alpha:
        conclusion = f"Rechazamos H0 al nivel de significancia {alpha}."
    else:
        conclusion = f"No rechazamos H0 al nivel de significancia {alpha}."

    return t_stat, p_value, conclusion

#%%
# Ejemplo de uso
sample = [50, 52, 48, 51, 49, 53, 50, 52, 49, 51, 50, 52, 51, 49, 50, 52, 51, 53, 50, 51, 49, 52, 53, 50, 51, 49, 52, 51, 50, 53]  # Datos muestrales (n = 30)
mu0 = 50                                # Media poblacional bajo H0
alternative = 'two-sided'               # Prueba bilateral
alpha = 0.05                            # Nivel de significancia

t_stat, p_value, conclusion = t_test_large_sample(sample, mu0, alternative, alpha)

print(f"Estadístico t: {t_stat:.4f}")
print(f"Valor-p: {p_value:.4f}")
print(f"Conclusión: {conclusion}")


#%%
