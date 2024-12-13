# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:45:37 2024

@author: Usuario
"""

import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt

from distfit import distfit

#%%

llegadas = pd.read_csv("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est\\proyectos\\restaurante\\llegadas_restaurante.csv")
platillos = pd.read_csv("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est\\proyectos\\restaurante\\platillos.csv")
bebidas = pd.read_csv("C:\\Users\\Usuario\\Documents\\scidata\\24_inf_est\\proyectos\\restaurante\\bebidas.csv")

#%%

llegadas.columns
llegadas.head(20)

#%%

llegadas["Hora_Llegada"]

llegadas["Tamaño_Grupo"]


#%%

##### Preprocesamiento

llegadas["hora_llegada"] = pd.to_datetime(llegadas["Hora_Llegada"],
                                          format='%H:%M')

#%%

llegadas["espera"] = llegadas["hora_llegada"].diff().dt.total_seconds()/60

#%%

ggplot(data=llegadas) + geom_histogram(mapping=aes(x="espera"),binwidth=5)

#%%%

################### Distribución de los tiempos de espera entre llegada y llegada

t_espera = llegadas["espera"][1:]


mi_modelo = distfit(todf=True)
mi_modelo.fit_transform(t_espera)
resultado = mi_modelo.summary

mi_modelo.plot_summary()
plt.show()

''' entonces los tiempos de espera entre llegadas es exponencial con scale=9.55'''

#%%

ggplot(data=llegadas) + geom_histogram(mapping=aes(x="Tamaño_Grupo"),binwidth=1)

#%%

freq_rel = llegadas["Tamaño_Grupo"].value_counts(normalize=True)

#%%

########## Simulación de un día de trabajo

tiempo = 0
total_llegadas = 0

while tiempo < 600:
    total_llegadas = total_llegadas + 1
    tiempo = tiempo + np.random.exponential(scale=9.55)
    
total_llegadas    

clientes = np.random.choice(freq_rel.index,total_llegadas,p=freq_rel.values)
len(clientes)


df_clientes = pd.DataFrame({"total":clientes})

ggplot(data=df_clientes) + geom_histogram(mapping=aes(x="total"),binwidth=1)


#%%

# bebidas
# platillos

pago_clientes = []

for n_cliente in clientes:
    pago_comida = np.random.choice(platillos["Precio"],n_cliente,replace=True).sum()
    pago_bebida = np.random.choice(bebidas["Precio"],n_cliente,replace=True).sum()
    pago_clientes.append(pago_comida+pago_bebida)

#%%

sum(pago_clientes)

#%%

######### Voy a crear una función que replique la simulación de un día de trabajo

def simulacion():
    tiempo = 0
    total_llegadas = 0

    while tiempo < 600:
        total_llegadas = total_llegadas + 1
        tiempo = tiempo + np.random.exponential(scale=9.55)
        
    clientes = np.random.choice(freq_rel.index,total_llegadas,p=freq_rel.values)
    pago_clientes = []

    for n_cliente in clientes:
        pago_comida = np.random.choice(platillos["Precio"],n_cliente,replace=True).sum()
        pago_bebida = np.random.choice(bebidas["Precio"],n_cliente,replace=True).sum()
        pago_clientes.append(pago_comida+pago_bebida)
    return sum(pago_clientes)


#%%

simulacion()

dias_simulados = 3000
sum([simulacion() for x in range(dias_simulados)]) / dias_simulados



    







#%%












