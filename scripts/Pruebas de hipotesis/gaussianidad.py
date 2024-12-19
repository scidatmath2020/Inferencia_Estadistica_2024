# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:20:55 2024

@author: Usuario
"""
import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt

import scipy.stats as stats
import random

# Para gráficos
import statsmodels.api as sm

# Para la pruebas de gaussianidad
from scipy.stats import shapiro, kstest, norm, anderson, jarque_bera

# Para que muestre todas las columnas de los dataframes
pd.set_option('display.max_columns', None)

#%%

########################## GENERACIÓN DE MUESTRAS

#generada con N(mu=3.5,sigma=2)
small_gauss = [2.9267007, 2.5763093, 4.9931801, 0.6564296, 1.4377333, 7.6412183, 2.9204735] 

#generada con t(3)
big_t = [-0.577103929579228, -0.0669625949987604, 0.123572935953355, -0.524985500797433, -1.23249669279686, 0.509597230395874, -0.729559305649031, -0.41684441016622, 1.28155478163868, 0.924508782035897, 0.827405247774813, 1.59785194962189, -1.47879497630707, -1.26201626124022, -0.0593983026205043, -0.178873361732746, 0.801185847793428, 0.333473064862654, 1.25186288055626, 2.35949695172828, -0.633493106081742, -1.05713142223298, 0.0212461334293823, 0.466063027431909, 0.0762121526958427, -0.843837287109611, -0.104022595760381, 5.78550093074697, 0.709799846598426, -0.0897824055310009, -0.999402655342385, 0.337761665033848, -0.0306307006025367, 1.47728344947859, -0.176164802725808, 0.690341335235668, -0.292183630229324, -0.844902899428558, -3.49551302890857, 1.43006662844371, 1.24850000914668, -0.180820066444685, -0.573485189819109, 0.349757398842014, -2.09754115696913, -0.352572352149588, -0.509125036161415, 0.712742491824159, 0.519051722042105, -3.00737218678664]

#generada con N(mu=5,sigma=1)
random.seed(2024)
big_gauss = stats.norm(scale=1, loc=5).rvs(1000)



# Tomado de https://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%205%20-%20Normality%20Testing.pdf
densidades_mexico = pd.DataFrame({
    'Region': [
        'Ajuno', 'Angahuan', 'Arantepacua', 'Aranza', 'Charapan', 'Cheran',
        'Cocucho', 'Comachuen', 'Corupo', 'Ihuatzio', 'Janitzio', 'Jaracuaro',
        'Nahuatzen', 'Nurio', 'Paracho', 'Patzcuaro', 'Pichataro',
        'Pomacuaran', 'Quinceo', 'Quiroga', 'San Felipe', 'San Lorenzo',
        'Sevina', 'Tingambato', 'Turicuaro', 'Tzintzuntzan', 'Urapicho'
    ],
    'Population_Density': [
        5.11, 5.15, 5.00, 4.13, 5.10, 5.22, 5.04, 5.25, 4.53, 5.74, 6.63, 5.73,
        4.77, 6.06, 4.82, 4.98, 5.36, 4.96, 5.94, 5.01, 4.10, 4.69, 4.97, 5.01,
        6.19, 4.67, 6.30
    ]
})

# Generada con lognorm con media e y desviación 0.5
np.random.seed(1)
mi_lognorm = stats.lognorm.rvs(s=.5, scale=np.exp(1),size=1000)

#%%

########################## HISTOGRAMAS DE MUESTRAS

plt.hist(small_gauss, edgecolor='black')
plt.show()

plt.hist(big_t, edgecolor='black')
plt.show()

plt.hist(big_gauss, edgecolor='black')
plt.show()

plt.hist(densidades_mexico.Population_Density, edgecolor='black')
plt.show()

plt.hist(mi_lognorm, edgecolor='black',bins=20)
plt.show()

#%%

########################## Graficos Q-Q

stats.probplot(small_gauss, dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.show()

stats.probplot(big_t, dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.show()

stats.probplot(big_gauss, dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.show()

stats.probplot(densidades_mexico.Population_Density, dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.show()

stats.probplot(mi_lognorm, dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.show()

#%%

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

#%%

pruebas_gauss, prueba_anderson = gaussian_test(densidades_mexico.Population_Density)

pruebas_gauss
prueba_anderson

#%%


