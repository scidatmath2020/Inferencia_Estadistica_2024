# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:05:53 2024

@author: Usuario
"""

'''
###############################################################################
###############################################################################
######### PRUEBA DE HIPÓTESIS PARA LA MEDIA CON VARIANZA CONOCIDA    ##########
#########                       POBLACION GENERAL                    ##########
#########                        MUESTRA GRANDE                      ##########
###############################################################################
###############################################################################
'''


from scipy.stats import norm
import numpy as np

def z_test_known_sigma(sample, mu0, sigma, alternative='two-sided', alpha=0.05):
    """
    Prueba de hipótesis Z para la media con desviación estándar conocida (muestra grande).
    Supone que la población no es necesariamente normal.

    Parameters:
    - sample: lista o arreglo con los datos muestrales.
    - mu0: media poblacional bajo la hipótesis nula (H0).
    - sigma: desviación estándar poblacional (conocida).
    - alternative: tipo de hipótesis ('two-sided', 'greater', 'less').
    - alpha: nivel de significancia (default: 0.05).

    Returns:
    - z_stat: valor del estadístico Z.
    - p_value: valor p de la prueba.
    - conclusion: decisión basada en el nivel de significancia.
    """
    n = len(sample)                        # Tamaño de la muestra
    x_bar = np.mean(sample)                # Media muestral
    z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))  # Estadístico Z

    # Cálculo del valor p según la hipótesis alternativa
    if alternative == 'two-sided':
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - norm.cdf(z_stat)
    elif alternative == 'less':
        p_value = norm.cdf(z_stat)
    else:
        raise ValueError("alternative debe ser 'two-sided', 'greater' o 'less'")

    # Decisión basada en el nivel de significancia
    if p_value < alpha:
        conclusion = f"Rechazamos H0 al nivel de significancia {alpha}."
    else:
        conclusion = f"No rechazamos H0 al nivel de significancia {alpha}."

    return z_stat, p_value, conclusion

#%%

# Ejemplo de uso
sample = [50, 52, 48, 51, 49, 53, 50, 52, 49, 51, 50, 52, 51, 49, 50, 52, 51, 53, 50, 51, 49, 52, 53, 50, 51, 49, 52, 51, 50, 53]  # Datos muestrales (n = 30)
mu0 = 50                               # Media poblacional bajo H0
sigma = 2                              # Desviación estándar poblacional conocida
alternative = 'two-sided'              # Prueba bilateral
alpha = 0.05                           # Nivel de significancia

z_stat, p_value, conclusion = z_test_known_sigma(sample, mu0, sigma, alternative, alpha)

print(f"Estadístico Z: {z_stat:.4f}")
print(f"Valor-p: {p_value:.4f}")
print(f"Conclusión: {conclusion}")
