import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn import preprocessing
import matplotlib.pyplot as plt

def ChoosingHousingFeatures(data,meanlimit,varlimit):
    features = []
    
    for column in data:
        normalizedcolumn = data[column]/data[column].max()
        if normalizedcolumn.mean()>meanlimit and normalizedcolumn.var() > varlimit:
            features.append(column)
    
    return features

def CleanHousingData(data):
    #Fix data errors
    data.loc[data['Exterior1st'] == 'Wd Shng', 'Exterior1st'] = 'WdShing'
    data.loc[data['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 'WdShing'
    data.loc[data['Exterior1st'] == 'Brk Cmn', 'Exterior1st'] = 'BrkComm'
    data.loc[data['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 'BrkComm'
    data.loc[data['Exterior1st'] == 'CmentBd', 'Exterior1st'] = 'CemntBd'
    data.loc[data['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 'CemntBd'

    features = list(data.columns[1:].values)
    

    strfeatures = []
    for feature in features:
        if data[feature].dtype == 'O' or data[feature].dtype == 'S' or data[feature].dtype == 'U' or data[feature].dtype == 'V':
            strfeatures.append(feature)


    Options = []
    for feature in strfeatures:
        Options.append([x for x in list(data[feature].unique()) if type(x) != float])

    for j in range(len(strfeatures)):
        for i in range(len(Options[j])):
            data.loc[data[strfeatures[j]] == Options[j][i],strfeatures[j]] = i     
    
    for feature in features:
        data[feature] = data[feature].fillna(int(data[feature].median()))
        nparray = data[feature].as_matrix()
        if np.isnan(np.any(nparray)) == True or np.isfinite(np.all(nparray)) == False:
            raise Exception('Couldn\'t Clean' + feature)
        

    return data

def NormalizeHousingData(data):
    data = np.log1p(data)
    
    return data

    '''   
    if skewing:
        features = list(data.columns.values)
        skewed_features = data[features].apply(lambda x: skew(x)) 
        skewed_features = skewed_features[skewed_features > 0.75]
        skewed_features = skewed_features.index
        data[skewed_features] = np.log1p(data[skewed_features])


    else:
        #normalizedsalespricevar = (data['SalePrice'] - data['SalePrice'].min())/(data['SalePrice'].max()-data['SalePrice'].min())
        #normalizedsalespricevar = normalizedsalespricevar.var()
        for column in data:
            if column != 'Id':
                    data[column] = (data[column] - data[column].min()+1)/(data[column].max()-data[column].min()+2)
    '''
def DeNormalizeHousingData(data):
    data = np.expm1(data)
    
    return data

    '''
    if skewing:
        features = list(data.columns[1:].values)
        skewed_features = data[features].apply(lambda x: skew(x)) 
        skewed_features = skewed_features[skewed_features > 0.75]
        skewed_features = skewed_features.index

    else:
        #normalizedsalespricevar = (data['SalePrice'] - data['SalePrice'].min())/(data['SalePrice'].max()-data['SalePrice'].min())
        #normalizedsalespricevar = normalizedsalespricevar.var()
        for column in data:
            if column != 'Id':
                    data[column] = (data[column] - data[column].min()+1)/(data[column].max()-data[column].min()+2)
    '''


if __name__ == "__main__":
    data = pd.read_csv('train.csv')
    cleaneddata = CleanHousingData(data)
    normalizeddata = NormalizeHousingData(cleaneddata)
    denormalized = DeNormalizeHousingData(normalizeddata)
    fig1 = plt.figure(figsize=(8, 3))
    cleaneddata['SalePrice'].hist()
    plt.title('Cleaned Data')
    fig2 = plt.figure(figsize=(8, 3))
    normalizeddata['SalePrice'].hist()
    plt.title('Normalized Data')
    features = ChoosingHousingFeatures(cleaneddata,0.001,0.01)
    print(features)