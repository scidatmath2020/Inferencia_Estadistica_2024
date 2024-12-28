# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:49:30 2024

@author: Usuario
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from plotnine import *
import itertools

#%%


muestra = pd.read_csv(r"C:\Users\Usuario\Documents\scidata\24_inf_est\proyectos\muestra_calidad_vida.csv")

Y = muestra['esp_vida']
X = muestra[['habitantes','ingresos','analfabetismo', 'asesinatos','universitarios','heladas','area','densidad_pobl']]

# Agregar una constante para el término de intersección
X = sm.add_constant(X)

# Ajustar el modelo de regresión lineal
modelo = sm.OLS(Y, X).fit()

#%%

modelo.summary()
modelo.rsquared
modelo.rsquared_adj

#%%

modelo.aic
modelo.bic

#%%

def stepwise_selection(data, target, predictors):
    """
    data: DataFrame con los datos.
    target: Nombre de la columna dependiente (Y).
    predictors: Lista de nombres de las columnas independientes (X).
    """
    best_aic = float('inf')
    best_model = None
    best_predictors = []

    # Generar todos los subconjuntos posibles de las variables independientes
    for k in range(1, len(predictors) + 1):
        for subset in itertools.combinations(predictors, k):
            X = sm.add_constant(data[list(subset)])  # Agregar el término constante
            Y = data[target]
            model = sm.OLS(Y, X).fit()
            aic = model.aic
            r2 = model.rsquared
            r2_adj = model.rsquared_adj
            
            if aic < best_aic:  # Comparar el AIC
                best_aic = aic
                best_model = model
                best_predictors = list(subset)
    
    return best_model, best_predictors

#%%
# Uso del paso anterior
predictors = ['habitantes',
              'ingresos',
              'analfabetismo',
              'asesinatos',
              'universitarios',
              'heladas',
              'area',
              'densidad_pobl']
target = 'esp_vida'
best_model, best_predictors = stepwise_selection(datos, target, predictors)

# Imprimir resultados
print(f"Mejor conjunto de predictores: {best_predictors}")
print(f"AIC del mejor modelo: {best_model.aic}")
print(best_model.summary())

best_model.conf_int()


#%%

# Nuevos datos para Alabama y Minnesota
new_data = pd.DataFrame({
    'ingresos': [3624, 4675],
    'asesinatos': [15.1, 2.3],
    'universitarios': [41.3, 57.6],
    'area': [50708, 79289]
})

# Agregar el término constante
new_data = sm.add_constant(new_data, has_constant='add')

# Predicciones con intervalos de confianza
prediccion = best_model.get_prediction(new_data).summary_frame(alpha=0.05)

# Mostrar resultados
print("Intervalo de confianza:")
print(prediccion[['mean', 'mean_ci_lower', 'mean_ci_upper']])

print("\nIntervalo de predicción:")
print(prediccion[['mean', 'obs_ci_lower', 'obs_ci_upper']])


