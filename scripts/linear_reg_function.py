# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:34:28 2024

@author: SciData
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from plotnine import *
import itertools
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, norm, anderson, jarque_bera,t, stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Para que muestre todas las columnas de los dataframes
pd.set_option('display.max_columns', None)

#%%

def linear_regression(data, target, predictors, best_ml = True):
    '''
    Nos devuelve un diagnóstico completo de una regresión lineal.
    
    data: nombre de la tabla
    target: nombre de la variable objetivo. Es un string
    predictors: lista de variables predictoras
    best_ml: Si True, indica que se busca el mejor modelo con las columnas predictors.
             Si False, indica que se busca el modelo lineal con exactamente las variables predictors.
             
    Devuelve una tupla de 21 objetos:
    modelo: modelo entrenado
    params: los coeficientes estimados b_i
    ci_params: los intervalos de confianza para los b_i 
    r2: R^2 del modelo (porcentaje de la varianza de Y explicada a través del modelo) 
    r2_adj: R^2 ajustado del modelo
    AIC: información de Akaike
    residuos: residuos del modelo
    residuos_qq: gráfico qq de los residuos
    residuos_hist: gráfico histograma de los residuos
    gaussianidad_residuos: resultados de la prueba de gaussianidad
    homocedasticidad_grafica: gráfico de las estimaciones del modelo vs residuos
    bp_p_value: p-valor del test de Breuch-Pagan para homocedasticidad de los residuos
    resultado_bptest: resultado del test de Breuch-Pagan para homocedasticidad de los residuos
    dw_stat: valor del estadístico de Durwin-Watson para identificar autocorrelación de los residuos
    resultado_dw: resultado del test de Durwin-Watson para identificar autocorrelación de los residuos
    p_value_tukey: p-valor del test de Tukey para aditividad
    resultado_pval_tukey: resultado del test de Tukey para aditividad
    residuos_parciales_grafica: gráfico de los residuos parciales para analizar linealidad
    indices_alto_leverage: índices de la tabla que presentan un alto apalancamiento
    indices_outliers_significativos: índices de la tabla que presentan un valor anómalo en la columna objetivo
    indices_alta_influencia: leverage alto y outlier. Afectan fuertemente a la regresión    
    predictores: columnas con las cuales se realizó la regresión final
    '''
    
    def gaussian_test(lista):
        shapiro_test = shapiro(lista)
        ks_test = kstest(lista,"norm",args=(np.mean(lista),np.std(lista,ddof=1)))
        jb_test = jarque_bera(lista)
        p_valores = [shapiro_test.pvalue, ks_test.pvalue,jb_test.pvalue] 
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
    
    if best_ml == True:
        modelo, predictores = stepwise_selection(data,target,predictors)
    else:
        X = sm.add_constant(data[predictors]) 
        y = data[target]
        modelo = sm.OLS(y,X).fit()
        predictores = predictors
    
    X = sm.add_constant(data[predictores])
    y = data[target]
    
    r2 = modelo.rsquared
    r2_adj = modelo.rsquared_adj
    AIC = modelo.aic
    
    n = len(data)  # Número de observaciones
    k = len(predictores)  # Número de predictores (excluye intercepto)
    
    residuos = modelo.resid
    
    gaussianidad_residuos = gaussian_test(residuos)
    tabla_residuos = pd.DataFrame({"value": residuos})
    
    residuos_qq = (ggplot(tabla_residuos, aes(sample="value")) +
        geom_qq() +  # Puntos del gráfico QQ
        geom_qq_line(color="red") +  # Línea de referencia en rojo
        theme(figure_size=(8, 6))
    )
    
    bins_sturges = int(np.ceil(np.log2(n) + 1)) 
    residuos_hist = (ggplot(data=tabla_residuos) +
                         geom_histogram(mapping=aes(x="value"),
                                        fill = "blue",
                                        color = "white",
                                        bins=bins_sturges)
                     )
    
    y_gorro = modelo.fittedvalues
    var_errores = pd.DataFrame({"y_gorro":y_gorro, "residuos":residuos})
    
    homocedasticidad_grafica = (ggplot(data=var_errores) + 
     geom_point(mapping=aes(x="y_gorro",y="residuos"),color="blue") +
     geom_hline(yintercept=0)
    )
    
    bp_test = het_breuschpagan(residuos, X)
    bp_stat = bp_test[0]   # Estadístico de la prueba
    bp_p_value = bp_test[1]  # p-valor asociado
    
    if bp_p_value > 0.05:
        resultado_bptest = "No se rechaza la hipótesis nula: los errores son homocedásticos."
    else:
        resultado_bptest = "Se rechaza la hipótesis nula: los errores no son homocedásticos."

    dw_stat = durbin_watson(residuos)
    if dw_stat < 1.5:
        resultado_dw = "Posible autocorrelación positiva en los residuos."
    elif dw_stat > 2.5:
        resultado_dw = "Posible autocorrelación negativa en los residuos."
    else:
        resultado_dw = "No hay evidencia significativa de autocorrelación en los residuos."
        
    X_copia = data.copy()
    X_copia["fitted"] = modelo.fittedvalues
    X_copia["fitted_squared"] = X_copia["fitted"]**2
    
    
    formula = f"{target} ~ " + " + ".join(predictores) + " + fitted_squared"
    modelo_tukey = smf.ols(formula, data=X_copia).fit()
    p_value_tukey = modelo_tukey.pvalues["fitted_squared"]
    
    if p_value_tukey < 0.05:
        resultado_pval_tukey = f"p-valor de Tukey: {p_value_tukey:.4f} - Rechazamos H0: el modelo NO es aditivo."
    else:
        resultado_pval_tukey = f"p-valor de Tukey: {p_value_tukey:.4f} - No Rechazamos H0: el modelo es aditivo."

    res_partial = [modelo.resid + modelo.params[x]*X.iloc[:,x] for x in range(1,k+1)]

    partial = pd.concat([pd.DataFrame({"Component+Residual(Y)":res_partial[x],
                   "type":predictores[x],
                   "X":X.iloc[:,x+1]}) for x in range(k)])

    residuos_parciales_grafica = (
    ggplot(data=partial) + 
        geom_point(mapping=aes(x="X",y="Component+Residual(Y)"),size=1,shape="o",fill="white") +
        geom_smooth(mapping=aes(x="X",y="Component+Residual(Y)"),method="lm",color="red",se=False) +
        facet_wrap("~type",scales="free_y",ncol=2) +
        theme(
            figure_size=(10, 6),
            plot_background=element_rect(fill="white"),
            panel_background=element_rect(fill="white"),
            panel_grid=element_line(color="gray", size=0.5),  # Cuadrícula de fondo
            panel_border=element_rect(color="black", size=0.5),  # Contorno rectangular de cada subgráfica
            strip_background=element_rect(fill="lightgray", color="black", size=0.5),  # Fondo y contorno del título de la faceta
            strip_text=element_text(color="black", size=12)  # Color y tamaño del texto del título de la faceta
        )
    )

    # Valores de apalancamiento
    influence = modelo.get_influence()
    hat_values = influence.hat_matrix_diag

    # Umbral para valores altos de apalancamiento
    high_leverage_threshold = 2 * (k + 1) / n
    high_leverage_points = np.where(hat_values > high_leverage_threshold)[0]
    indices_alto_leverage = high_leverage_points
    
    # Residuos estudiantizados
    studentized_residuals = influence.resid_studentized_external
    p_values_outliers = 2 * (1 - t.cdf(np.abs(studentized_residuals), df=n - k - 1))

    # Filtrar outliers significativos (ejemplo con p < 0.05)
    outliers = np.where(p_values_outliers < 0.05)[0]
    indices_outliers_significativos = outliers
    
    # Distancia de Cook
    cooks_d = influence.cooks_distance[0]

    # Umbral para alta influencia
    cooks_threshold = 4 / (n - k - 2)
    high_influence_points = np.where(cooks_d > cooks_threshold)[0]
    indices_alta_influencia = high_influence_points
    
    params = modelo.params
    ci_params = modelo.conf_int()
    
    return modelo, params, ci_params, r2, r2_adj, AIC, residuos, residuos_qq, residuos_hist, gaussianidad_residuos, homocedasticidad_grafica, bp_p_value, resultado_bptest, dw_stat, resultado_dw, p_value_tukey, resultado_pval_tukey, residuos_parciales_grafica, indices_alto_leverage, indices_outliers_significativos, indices_alta_influencia, predictores



