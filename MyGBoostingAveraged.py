# -*- coding: utf-8 -*-
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:38:58 2017

@author: bonny

Investigate how to calculate the core of a regression problem
"""
from sklearn.metrics import explained_variance_score, mean_absolute_error
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model, ensemble
import numpy as np
from CleanHousingData import CleanHousingData,NormalizeHousingData,DeNormalizeHousingData, ChoosingHousingFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#Read the train file
tr_data = pd.read_csv('train.csv')

#Print head and dexcription
#print(tr_data.head())
#print(tr_data.describe())


#Clean the data
tr_data = CleanHousingData(tr_data)

features = list(tr_data.columns[1:].values)
features.remove('SalePrice')

#Split the data with the folds
kf = KFold(n_splits=3, random_state=1, shuffle=True)
for train_index, test_index in kf.split(tr_data):
    trainsplit = tr_data.iloc[train_index,:] 
    testsplit =  tr_data.iloc[test_index,:] 

Ntrainsplit = NormalizeHousingData(trainsplit)
Ntestsplit = NormalizeHousingData(testsplit)
#Finding out which algorithm adjusts better to the data
#Create the algorithm dictionary
LR= linear_model.LinearRegression()
Gboost = ensemble.GradientBoostingRegressor()
algorithms = [ Gboost, LR]
names = ['Gradient Boosting', 'Linear Regression']

predictors = ChoosingHousingFeatures(tr_data,0.001,0.0005)
predictors.remove('SalePrice')
allOutputs = []

for i in range(len(algorithms)):
    algorithms[i].fit(Ntrainsplit[predictors],Ntrainsplit['SalePrice'])
    NOutput = algorithms[i].predict(Ntestsplit[predictors])
    Output = DeNormalizeHousingData(NOutput)
    allOutputs.append(Output)
    # The mean squared error
    meanerror = mean_absolute_error(testsplit['SalePrice'],Output)
    print(names[i] + ' ' +  " Mean squared error: %.3f" % meanerror)
    # Explained variance score: 1 is perfect prediction
    varscore = explained_variance_score(testsplit['SalePrice'], Output)
    print(names[i] + ' ' + ' Variance score: %.3f' % varscore)



#Average
prediction = np.zeros(len(allOutputs[0]))
for i in range(len(allOutputs[0])):
    prediction[i] = (4*allOutputs[0][i] + allOutputs[1][i])/5

Mean = mean_absolute_error(testsplit['SalePrice'],prediction)
print('Mean error: %f' % Mean) 
Score = explained_variance_score(testsplit['SalePrice'],prediction)
print('Average score: %f' % Score)  
 

#Test and Submission
test_data = pd.read_csv('test.csv')
test_data = CleanHousingData(test_data)
Ntest_data = NormalizeHousingData(test_data)

alltestOutputs = []
for i in range(len(algorithms)):
    NOutput = algorithms[i].predict(Ntest_data[predictors])
    Output = DeNormalizeHousingData(NOutput)
    alltestOutputs.append(Output)

test_predictions = np.zeros(len(alltestOutputs[0]))
for i in range(len(alltestOutputs[0])):
    test_predictions[i] = (4*alltestOutputs[0][i] + alltestOutputs[1][i])/5

submission = pd.DataFrame({
        "Id": test_data["Id"],
        "SalePrice": test_predictions
    })
    
submission.to_csv('./GboostAveragedSubmission.csv',index=False)
