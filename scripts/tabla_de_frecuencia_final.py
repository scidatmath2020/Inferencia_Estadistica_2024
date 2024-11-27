# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:14:03 2024

@author: Usuario
"""

import pandas as pd
import numpy as np
import os

os.chdir("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est\\proyectos\\calificaciones")
os.getcwd() 

#%%

'''Lectura de la tabla de datos'''

# Leer el archivo CSV
df = pd.read_csv('calif_estudiantes.csv')
df
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

### Regla de Sturges
c = round(1 + np.log2(df.shape[0])) 
bins = np.linspace(min(df[columna]), max(df[columna]),c+1)

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

# Guardar la tabla. Entre comillas va el nombre con el cual quieres guardar
# el archivo en tu computadora. No olvides el .csv y que va entre comillas

table.to_csv("frecuencias_relativas_sturges.csv",index=False)



