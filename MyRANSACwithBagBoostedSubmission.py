# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Thu Jun  1 14:38:58 2017

@author: bonny
"""
from sklearn.metrics import explained_variance_score, mean_absolute_error
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model, ensemble
import numpy as np
from CleanHousingData import CleanHousingData,NormalizeHousingData,DeNormalizeHousingData
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

Ntrainsplit = NormalizeHousingData(trainsplit)
Ntestsplit = NormalizeHousingData(testsplit)

#Finding out which algorithm adjusts better to the data
#Create the algorithm dictionary
alg = linear_model.RANSACRegressor()
booster = ensemble.BaggingRegressor
SuperAlg = booster(base_estimator =alg)
SuperAlg.fit(Ntrainsplit[predictors],Ntrainsplit['SalePrice'])
NOutput = SuperAlg.predict(Ntestsplit[predictors])

print('algorithm score: %.3f' % SuperAlg.score(testsplit[predictors],testsplit['SalePrice']))

Output = DeNormalizeHousingData(NOutput)
assert np.isnan(np.any(Output)) == False and np.isfinite(np.all(Output)) == True, "bad Value"
assert np.isnan(np.any(testsplit['SalePrice'])) == False and np.isfinite(np.all(testsplit['SalePrice'])) == True, "bad Value"
# The mean squared error
meanerror = mean_absolute_error(testsplit['SalePrice'],Output)
print(" Mean squared error: %.3f" % meanerror)
# Explained variance score: 1 is perfect prediction
varscore = explained_variance_score(testsplit['SalePrice'],Output)
print(' Variance score: %.3f' % varscore)

           
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(Output, testsplit['SalePrice'], color='black')
ax.plot(Output, testsplit['SalePrice'], 'orange') 
plt.title('RANSACRegressor with Bag Boost')
plt.xticks(())
plt.yticks(())
ax.set_zticks(())
plt.show()

#Test and Submission
test_data = pd.read_csv('test.csv')
test_data = CleanHousingData(test_data)
Ntest_data = NormalizeHousingData(test_data)
Ntest_predictions = SuperAlg.predict(Ntest_data[predictors])
test_predictions = DeNormalizeHousingData(Ntest_predictions)

submission = pd.DataFrame({
        "Id": test_data["Id"],
        "SalePrice": test_predictions
    })
    
submission.to_csv('./RANSACBagBoostedSubmission.csv',index = False)