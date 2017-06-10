#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:38:58 2017

@author: bonny
"""
import signal

def signal_handler(signum, frame):
    raise Exception("Timed out!")

signal.signal(signal.SIGALRM, signal_handler)


import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model, ensemble
import numpy as np
from CleanHousingData import CleanHousingData,NormalizeHousingData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Read the train file
tr_data = pd.read_csv('train.csv')

#Print head and dexcription
#print(tr_data.head())
#print(tr_data.describe())

features = list(tr_data.columns[1:].values)
features.remove('SalePrice')

#Choose predictors
predictors = features

#Clean the data
tr_data = CleanHousingData(tr_data)

#Split the data with the folds
kf = KFold(n_splits=3, random_state=1, shuffle=True)
for train_index, test_index in kf.split(tr_data):
    trainsplit = tr_data.iloc[train_index,:] 
    testsplit =  tr_data.iloc[test_index,:] 

#Finding out which algorithm adjusts better to the data
#Create the algorithm dictionary
ARD= linear_model.ARDRegression()
LinRe = linear_model.LinearRegression()
SGD = linear_model.SGDRegressor()
BR= linear_model.BayesianRidge()
Lars = linear_model.Lars()
Lasso = linear_model.Lasso()
PA = linear_model.PassiveAggressiveRegressor()
RANSAC = linear_model.RANSACRegressor()
Theil = linear_model.TheilSenRegressor()
Gboost = ensemble.GradientBoostingRegressor()
algorithms = {'Linear Regression':LinRe, 'Bayesian ARD regression':ARD, 
              'BayesianRidge': BR,'Lars': Lars,'Lasso':Lasso, 
              'PassiveAggressiveRegressor':PA ,'RANSACRegressor':RANSAC,
              'GradientBoostingRegressor':Gboost}
Ada = ensemble.AdaBoostRegressor
Bag = ensemble.BaggingRegressor
boosters = {'AdaBoost':Ada, 'Bagging Regressor':Bag}

color = {'Linear Regression':'blue','Bayesian ARD regression':'red', 
         'BayesianRidge':'yellow', 'Lars': 'green','Lasso': 'orange'
         ,'PassiveAggressiveRegressor':'purple','RANSACRegressor':'pink',
         'GradientBoostingRegressor':'cyan'}

algstrings=['Linear Regression', 'Bayesian ARD regression', 
              'BayesianRidge','Lars','Lasso', 
              'PassiveAggressiveRegressor' ,'RANSACRegressor',
              'GradientBoostingRegressor']
boosterstrings = ['AdaBoost', 'Bagging Regressor']

oldvarscore = 0
oldmeanerror = float('inf')
bestalg = 0
secondbestalg = 0
BestOutput = 0
for booster in boosters:
    for alg in algorithms:
        signal.alarm(60)   # Ten seconds
        try:
            SuperAlg = boosters[booster](base_estimator =algorithms[alg])
            SuperAlg.fit(trainsplit[predictors],trainsplit['SalePrice'])
            Output = SuperAlg.predict(testsplit[predictors])
            # The mean squared error
            meanerror = np.mean((Output - testsplit['SalePrice']) ** 2)
            print(alg + ' ' +  booster + " Mean squared error: %.3f" % meanerror)
            # Explained variance score: 1 is perfect prediction
            varscore = SuperAlg.score(testsplit[predictors], testsplit['SalePrice'])
            print(alg + ' ' + booster + ' Variance score: %.3f' % varscore)
            if varscore > oldvarscore:
                BestOutput = Output
                secondbestalg = bestalg
                bestalg = alg
                bestbooster = booster
                oldvarscore = varscore
                oldmeanerror = meanerror
            elif varscore == oldvarscore and meanerror < oldmeanerror:
                BestOutput = Output
                secondbestalg = bestalg
                bestalg = alg
                bestbooster = booster
                oldvarscore = varscore
                oldmeanerror = meanerror
        except:
            print('Timed out with:' + booster+ 'and ' + alg)

print("Best fitted algorithm is: " + bestalg + "with " + bestbooster )           
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(BestOutput, testsplit['SalePrice'], color='black')
ax.plot(BestOutput, testsplit['SalePrice'], color[bestalg]) 
plt.title(bestalg)
plt.xticks(())
plt.yticks(())
ax.set_zticks(())
plt.show()

