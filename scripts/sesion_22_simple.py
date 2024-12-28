# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:01:19 2024

@author: Usuario
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotnine import *

pd.set_option('display.max_columns', None)


#%%
# Crear el DataFrame
presion = pd.DataFrame({
    "sal": [1.8, 2.2, 3.5, 4.0, 4.3, 5.0],
    "tension": [100, 98, 110, 110, 112, 120]
})

#%%
(ggplot(data=presion) +
     geom_smooth(mapping=aes(x="sal",y="tension"),method="lm",color="red") +
     geom_point(mapping=aes(x="sal",y="tension"), color="blue")
     )

#%%
# Agregar una constante para la intercepción
X = sm.add_constant(presion["sal"])
y = presion["tension"]

#%%
# Realizar la regresión lineal
modelo = sm.OLS(y, X).fit()

# Obtener los coeficientes
coeficientes = modelo.params
modelo.summary()

#%%
r_squared = modelo.rsquared
print(r_squared)


#%%

modelo.conf_int(alpha=0.95)

#%%

# Nuevo dato para predicción
new_data = pd.DataFrame({'sal': [4.5]})
new_data = sm.add_constant(new_data, has_constant='add')  # Asegurar que la constante esté incluida

# Predicción con intervalo de confianza
prediccion = modelo.get_prediction(new_data).summary_frame(alpha=0.05)  # 95% confidence interval

# Mostrar resultados
print("Intervalo de confianza:")
print(prediccion[['mean', 'mean_ci_lower', 'mean_ci_upper']])

print("\nIntervalo de predicción:")
print(prediccion[['mean', 'obs_ci_lower', 'obs_ci_upper']])
