# -*- coding: utf-8 -*-
"""
Created on Fri Jan 03 00:41:33 2023

@author: Gonzalo
"""


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

variables = ["D_viento","H_relativa","P_atmosferica",
                        "Temperatura","V_viento","Pluviometria","Media_D_viento",
                        "Media_V_viento","Media_P_atmosferica","Media_Temperatura",
                        "Media_H_relativa","Minimo_Temperatura","Maximo_Temperatura",
                        "Rango_Temperatura","Media_PM2_5","Minimo_PM2_5","Maximo_PM2_5",
                        "Rango_PM2_5","PM10"]


"""
#Carga del dataset la florida para PM10 imputado por KNN
PM_pandas=pd.read_csv('./horarioAcotado/lf_pm25a.csv')

#Se asignan datos inferiores a 2018 para entranamiento
training = PM_pandas.loc[PM_pandas.FECHA <np.int64(180000)]
#Se asignan datos superiores a 2018 para test
test =  PM_pandas.loc[PM_pandas.FECHA >=180000]

print("# Tuning hyper-parameters" )
print()


gsc = GridSearchCV(
    estimator=SVR(kernel='rbf'),
    param_grid={
        'C': [5,10,20,30,40,50,100,200,300,345,347,350,400],
        'epsilon': [0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.075,0.1],
        'gamma': [0.00001,0.00005,0.0001,0.0005,0.001]
    },
cv=10, scoring='r2', verbose=0, n_jobs=30)

fold= TimeSeriesSplit(max_train_size=None, n_splits=5)
for train_index, test_index in fold.split(training):
    #Se asignan variables base sin medias ni min. ni max.
    X_train = training[variables].iloc[train_index].values
    grid_result = gsc.fit(X_train,training["PM2_5"].iloc[train_index].values)
    #Se asignan variables base sin medias ni min. ni max.
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.0, shrinking=True,
                   tol=1e-3, cache_size=200, verbose=False, max_iter=-1)
    print(best_svr)

print("mejor_svr: ",best_svr)

"""
######################### CALCULO PARA PM2_5 ###################################
print("\n***********************************************************")
print("**************** Analisis de dataset PM2_5 *****************")
print("***********************************************************\n")

#Carga del dataset completos la florida, ucm y utal, para PM2_5 imputado por KNN
lf=pd.read_csv('./lf_pm25_ia.csv')

#Carga del dataset acotados la florida, ucm y utal, para PM2_5 imputado por KNN
lf_acot=pd.read_csv('./horarioAcotado/lf_pm25a.csv')

#Se asignan datos inferiores a 2018 para entranamiento
training1 = lf.loc[lf.FECHA <np.int64(180000)]
#Se asignan datos superiores a 2018 para test
test_lf =  lf.loc[lf.FECHA >=180000]
#SVR_PM2_5 = best_svr
#SVR_PM2_5 = SVR(C=450, epsilon=0.005, gamma=0.000009)
SVR_PM2_5 = SVR(C=50, epsilon=0.0075, gamma=1e-05)
print(SVR_PM2_5)


#********************************** DATA BASICO *****************************************************#

################### La florida ######################
#Validacion simple con K-fold de 10 pasos
#fold= KFold(n_splits=5, shuffle=True, random_state=0)

fold= TimeSeriesSplit(max_train_size=None, n_splits=10)
for train_index, test_index in fold.split(training1):
    #Se asignan variables extendidas
    X_train = training1[variables].iloc[train_index].values
    
    X_test =  training1[variables].iloc[test_index].values
    
    SVR_PM2_5.fit(X_train,training1["PM2_5"].iloc[train_index].values)

metrica_train1 = []
predict_PM2_5 = SVR_PM2_5.predict(training1[variables].values)
metrica_train1.append(["PM2_5","R2",r2_score(training1["PM2_5"].values, predict_PM2_5)])


print("\n********** Metricas de entrenamiento LA florida completo basico**********\n")
print(metrica_train1)

lista = SVR_PM2_5.predict(test_lf[variables].values)
print(["PM2_5","R2",r2_score(test_lf["PM2_5"].values, lista)])

dfNuevo.round(3).to_csv('./predichos 2018.csv',index = False, sep=';',decimal=',')
dfNuevo = pd.DataFrame()
dfNuevo['pred'] = lista

