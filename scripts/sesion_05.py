# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:14:01 2024

@author: Usuario
"""

### numpy, pandas

#### importado de paqueterías

#### libro.lo_que_necesite
import numpy as np
import pandas as pd
import math
from plotnine import *
import os

#%%
##### Elegir directorio (carpeta) de trabajo
os.chdir("C:/Users/Usuario/Documents/scidata/24_inf_est/proyectos/sismos")

##### lectura
tabla = pd.read_csv("sismos_procesado.csv",encoding="latin1")

#### encabezado
tabla.head()

#### nombres de las columnas
tabla.columns

#### cuántas filas y columnas tiene
tabla.shape

#### resumen estadístico de las columnas numéricas
tabla.describe()


#%%

math.sin(4.5)

np.cos([1,5,3,4])

math.cos(86523)

math.cos([1,5,3,4])

#%%

data_prueba = pd.DataFrame({"hola":["a","b","a"],
                            "col2":[1.5,2,1.6],
                            "col3":[1,2,3]
    })
data_prueba

(ggplot(data=data_prueba) + 
 geom_point(mapping=aes(x="col2",y="col3",color="col1"))
)


mi_grafica.show()

#%%




