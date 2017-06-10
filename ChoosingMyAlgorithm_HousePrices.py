#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:38:58 2017

@author: bonny
"""
from sklearn.metrics import explained_variance_score
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

#Clean the data
tr_data = CleanHousingData(tr_data)


#Choose predictors
predictors = list(tr_data.columns[1:].values)
predictors.remove('SalePrice')

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
Gboost = ensemble.GradientBoostingRegressor()
algorithms = {'Linear Regression':LinRe, 
              #'Bayesian ARD regression':ARD, 
              'BayesianRidge': BR,'Lars': Lars,
              #'Lasso':Lasso, 
              #'PassiveAggressiveRegressor':PA ,
              #'RANSACRegressor':RANSAC,
              'GradientBoostingRegressor':Gboost
              }

color = {'Linear Regression':'blue','Bayesian ARD regression':'red', 
         'BayesianRidge':'yellow', 'Lars': 'green','Lasso': 'orange'
         ,'PassiveAggressiveRegressor':'purple','RANSACRegressor':'brown',
         'GradientBoostingRegressor':'cyan'}

bestvarscore = 0
bestmeanerror = float('inf')
secondbestvarscore = 0
secondbestmeanerror = float('inf')
bestalg = 0
secondbestalg = 0
SecondBestOutput = 0
BestOutput = 0
allpredictions = []
for alg in algorithms:
    algorithms[alg].fit(trainsplit[predictors],trainsplit['SalePrice'])
    Output = algorithms[alg].predict(testsplit[predictors])
    allpredictions.append(Output)
    # The mean squared error
    meanerror = np.mean((Output - testsplit['SalePrice']) ** 2)
    print(alg + ' ' +  " Mean squared error: %.3f" % meanerror)
    # Explained variance score: 1 is perfect prediction
    varscore = algorithms[alg].score(testsplit[predictors], testsplit['SalePrice'])
    print(alg + ' ' + ' Variance score: %.3f' % varscore)
    if varscore > bestvarscore:
        SecondBestOutput = BestOutput
        BestOutput = Output
        secondbestalg = bestalg
        bestalg = alg
        secondbestvarscore = bestvarscore
        bestvarscore = varscore
        secondbestmeanerror = bestmeanerror
        bestmeanerror = meanerror
        
    elif varscore == bestvarscore and meanerror < bestmeanerror:
        SecondBestOutput = BestOutput
        BestOutput = Output
        secondbestalg = bestalg
        bestalg = alg
        secondbestvarscore = varscore
        bestvarscore = varscore
        secondbestmeanerror = bestmeanerror
        bestmeanerror = meanerror
        
    elif varscore > secondbestvarscore:
        SecondBestOutput = Output
        secondbestalg = alg
        secondbestvarscore = varscore
        secondbestmeanerror = bestmeanerror
    
    elif varscore == secondbestvarscore and meanerror < secondbestmeanerror:
        SecondBestOutput = Output
        secondbestalg = alg
        secondbestvarscore = varscore
        secondbestmeanerror = bestmeanerror
        

print("Best fitted algorithm is: " + bestalg + " with %.3f " % bestvarscore + 'of variance score' )                  
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(BestOutput, testsplit['SalePrice'], color='black')
ax.plot(BestOutput, testsplit['SalePrice'], color[bestalg]) 
plt.title(bestalg)
plt.xticks(())
plt.yticks(())
ax.set_zticks(())
plt.show()
print("Second Best fitted algorithm is: " + secondbestalg + " with %.3f " % secondbestvarscore + 'of variance score')                  
fig = plt.figure()
ax = fig.gca(projection='3d')
ax = fig.gca()
ax.scatter(BestOutput, testsplit['SalePrice'], color='black')
ax.plot(BestOutput, testsplit['SalePrice'], color[secondbestalg]) 
plt.title(secondbestalg)
plt.xticks(())
plt.yticks(())
ax.set_zticks(())
plt.show()

averageprediction = sum(allpredictions)/len(allpredictions)

AverageScore =  explained_variance_score(testsplit['SalePrice'],averageprediction)

print('Average: %.3f' % AverageScore)