# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:09:58 2024

@author: Usuario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import pingouin as pg

import scipy.stats as stats
import random

# Para la pruebas de gaussianidad
from scipy.stats import shapiro, kstest, norm, anderson, jarque_bera

# Para que muestre todas las columnas de los dataframes
pd.set_option('display.max_columns', None)



#%%

beisbol = pd.read_csv(r"C:\Users\Usuario\Documents\scidata\24_inf_est\proyectos\beisbol.csv")
beisbol.head()
beisbol.shape


#%%

beisbol.groupby('posicion').size()
beisbol.groupby('posicion').agg(['mean', 'std'])

(ggplot(data=beisbol) + geom_boxplot(mapping=aes(x="posicion",y="bateo")))

#%%

### Paso 1: gaussianidad

def gaussian_test(lista):
    shapiro_test = shapiro(lista)
    ks_test = kstest(lista,"norm",args=(np.mean(lista),np.std(lista,ddof=1)))
    jb_test = jarque_bera(lista)
    p_valores = [shapiro_test.pvalue, ks_test.pvalue,jb_test.pvalue] 
#    previos = [x<0.05 for x in p_valores]
    rech = ["Rechazar H0","No rechazar H0"]
    significados = ["Hay buena probabilidad de que NO es gaussiana",
                    "Hay buena probabilidad de que SÍ es gaussiana"]
        
    resultados = pd.DataFrame({"Prueba":["SW","KS","JB"],
     "p_valor":p_valores,
     "Resultado":[rech[0] if x<0.05 else rech[1] for x in p_valores],
     "Interpretación":[significados[0] if x<0.05 else significados[1] for x in p_valores]
    })
    
    anderson_test = anderson(lista,dist="norm")
    est_A = anderson_test.statistic
    resultados_anderson = pd.DataFrame({"Estadístico A":5*[est_A],#,"","","","",""],
                                        "Significancia":anderson_test.significance_level,
                                        "Valor_crítico":anderson_test.critical_values,
                                        "Resultado":[rech[0] if est_A > anderson_test.critical_values[i] else rech[1] for i in range(5)],
                                        "Interpretación":[significados[0] if est_A > anderson_test.critical_values[i] else significados[1] for i in range(5)]})
    return  resultados, resultados_anderson

resultados_gaussinadad = beisbol.groupby('posicion')['bateo'].apply(gaussian_test)

resultados_gaussinadad[0]
resultados_gaussinadad[1]
resultados_gaussinadad[2]
resultados_gaussinadad[3]

(
ggplot(data = beisbol) + geom_histogram(mapping=aes(x="bateo")) +
    facet_wrap("~posicion")
)

#%%
#### Paso 2. Homocedasticidad

pg.homoscedasticity(data=beisbol, 
                    dv='bateo', 
                    group='posicion',
                    method='levene')

#%%

#### Paso 3. ANOVA

pg.anova(data=beisbol, 
         dv='bateo',
         between='posicion', 
         detailed=True)

#%%

#### Paso 4. Análisis post-hoc

pg.pairwise_tests(data=beisbol, 
                  dv='bateo', 
                  between='posicion', 
                  padjust='bonferroni')


pg.pairwise_tests(data=beisbol, 
                  dv='bateo', 
                  between='posicion', 
                  padjust='holm')


pg.pairwise_tukey(data=beisbol, 
                  dv='bateo', 
                  between='posicion')

pg.pairwise_gameshowell(data=beisbol, 
                        dv='bateo',
                        between='posicion')

