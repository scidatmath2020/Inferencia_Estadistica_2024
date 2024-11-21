# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:52:20 2024

@author: Usuario
"""

import pandas as pd
import numpy as np
import os

os.chdir("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est")

#%%

'''Lectura de la tabla de datos'''

# Leer el archivo CSV
df = pd.read_csv('calif_estudiantes.csv')

# Mostrar el nombre de las columnas
df.columns

# Mostrar las primeras 5 filas de la tabla
df.head()

# Columna de nuestro interés. En este caso es la columna exam_score
columna = "exam_score"

#%%

'''Personalización de la tabla de frecuencias'''

# Crear los intervalos (clases)
bins = [0, 20, 40, 60, 80, 100]

#%%
# Crear los nombres de las clases 
labels = [f"{bins[x]}-{bins[x+1]}"  for x in range(0,len(bins)-1)]

#%%

df['Class'] = pd.cut(df[columna], bins=bins, labels=labels, right=False)

# Crear las marcas de clase
class_value = [(bins[x]+bins[x+1])/2 for x in range(0,len(bins)-1)]

# Calcular la frecuencia
frequency = df['Class'].value_counts(sort=False)

# Calcular la frecuencia relativa
relative_frequency = frequency / frequency.sum()

# Calcular la frecuencia relativa acumulada
cumulative_relative_frequency = relative_frequency.cumsum()

# Crear la tabla final
table = pd.DataFrame({
    'Class': labels,
    'Class_value': class_value,
    'Frequency': frequency.values,
    'Relative_frequency': relative_frequency.values,
    'Cumulative_relative_frequency': cumulative_relative_frequency.values
})

# Mostrar la tabla
table

#%%

# Guardar la tabla

table.to_csv("frecuencias_relativas.csv",index=False)





