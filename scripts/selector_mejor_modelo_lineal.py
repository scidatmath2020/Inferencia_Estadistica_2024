# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:58:11 2024

@author: Usuario
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from plotnine import *
import itertools

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