
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
File created by Tyler McCandless for the NASA-AIST Wildfire Project.

"""
# Imports Required
# Pandas is used for data manipulation, h5py for data (conda install h5py)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
#import pydot
#import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import h5py
import numpy.ma as ma

# Load in HDF5 data
#store = pd.read_hdf('/glade/p/ral/wsap/NASA-FMC/sample_dfmc.2.h5','samples')
#storeAll.close()
storeAll = pd.HDFStore('/glade/p/ral/wsap/NASA-FMC/co_dfmc.h5')
store = storeAll.select('samples',where=('distNearestRaws < 2500','distNearestRaws > -9999.0'))
store = store.replace(-9999.0,np.NaN).dropna(axis=0,how='any')

# As example, set-up one random forest to train on
dFMC = store['fuelMoisture']
#index = store['index']
distNearestRaws = store['distNearestRaws']
ndvi = store['ndvi']
ndwi = store['ndwi']
gvmi = store['gvmi']
pmi = store['pmi']
vari = store['vari']
b1 = store['one_km_Surface_Reflectance_Band_1']
b2 = store['one_km_Surface_Reflectance_Band_2']
b3 = store['one_km_Surface_Reflectance_Band_3']
b4 = store['one_km_Surface_Reflectance_Band_4']
b5 = store['one_km_Surface_Reflectance_Band_5']
b6 = store['one_km_Surface_Reflectance_Band_6']
b7 = store['one_km_Surface_Reflectance_Band_7']
Ind = np.array([ndvi,ndwi,gvmi,pmi,vari,dFMC])
Bands = np.array([b1,b2,b3,b4,b5,b6,b7,dFMC])
featuresI = ['ndvi','ndwi','gvmi','pmi','vari']
featuresB = ['b1','b2','b3','b4','b5','b6','b7']
Ind = Ind.transpose()
Bands = Bands.transpose()
print(Ind.shape)
print(Bands.shape)
predictorsI = Ind[:,0:5]
predictandI = Ind[:,5]
predictorsB = Bands[:,0:7]
predictandB = Bands[:,7]

# Test adding in GVMI as a predictor with the reflectances
featuresGB = ['b1','b2','b3','b4','b5','b6','b7','gvmi']
GBands = np.array([b1,b2,b3,b4,b5,b6,b7,gvmi,dFMC])
GBands = GBands.transpose()
predictorsGB = GBands[:,0:8]
predictandGB = GBands[:,8]

# Random forest method
# Split the data into training and testing sets
train_featuresI, test_featuresI, train_labelsI, test_labelsI = train_test_split(predictorsI, predictandI, 
                 test_size = 0.20, random_state = 42)
train_featuresB, test_featuresB, train_labelsB, test_labelsB = train_test_split(predictorsB, predictandB, test_size = 0.20, random_state = 42)
train_featuresGB, test_featuresGB, train_labelsGB, test_labelsGB = train_test_split(predictorsGB, predictandGB, test_size = 0.20, random_state = 42)
# Section below to understand the data after separating into training and testing datasets
print('Training Features Shape:', train_featuresI.shape)
print('Training Labels Shape:', train_labelsI.shape)
print('Testing Features Shape:', test_featuresI.shape)
print('Testing Labels Shape:', test_labelsI.shape)

######### Configure and train the Random Forest ##############
# Import the Random Forest Model
# Instantiate model 
rfI = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)
rfB = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)
rfGB = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)

# Train the model on training data
rfI.fit(train_featuresI, train_labelsI);
rfB.fit(train_featuresB, train_labelsB);
rfGB.fit(train_featuresGB, train_labelsGB);

# Use the forest's predict method on the test data
predictionsI = rfI.predict(test_featuresI)
predictionsB = rfB.predict(test_featuresB)
predictionsGB = rfGB.predict(test_featuresGB)

######### Compute Errors on Test Data #############
# Calculate the absolute errors
errorsI = abs(predictionsI - test_labelsI)
errorsB = abs(predictionsB - test_labelsB)
errorsGB = abs(predictionsGB - test_labelsGB)
# Print out the mean absolute error (mae)
print('Mean Absolute Error with Indices:', round(np.mean(errorsI), 2))
print('Mean Absolute Error with Reflectance Bands:', round(np.mean(errorsB), 2))
print('Mean Absolute Error with GVMI & Reflectance Bands:', round(np.mean(errorsGB), 2))

# Feature selection
importancesI = list(rfI.feature_importances_)
importancesB = list(rfB.feature_importances_)
importancesGB = list(rfGB.feature_importances_)
feature_listI = featuresI
feature_listB = featuresB
feature_listGB = featuresGB
feature_importancesI = [(featuresI,round(importancesI,2)) for featuresI, importancesI in zip(feature_listI,importancesI)]
feature_importancesB = [(featuresB,round(importancesB,2)) for featuresB, importancesB in zip(feature_listB,importancesB)]
feature_importancesGB = [(featuresGB,round(importancesGB,2)) for featuresGB, importancesGB in zip(feature_listB,importancesGB)]
feature_importancesI = sorted(feature_importancesI,key = lambda x: x[1],reverse=True)
feature_importancesB = sorted(feature_importancesB,key = lambda x: x[1],reverse=True)
feature_importancesGB = sorted(feature_importancesGB,key = lambda x: x[1],reverse=True)
[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesI];
print('...and for reflectances...')
[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesB];
print('...and for GVMI & reflectances...')
[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesGB];


plt.scatter(rfGB.predict(train_featuresGB),rfGB.predict(train_featuresGB)-train_labelsGB,c='b',s=40,alpha=0.5)
plt.scatter(rfGB.predict(test_featuresGB),rfGB.predict(test_featuresGB)-test_labelsGB,c='g',s=40)
plt.title('Residual Plot using training (blue) and test (green) data',fontsize=18)
plt.ylabel('Residuals',fontsize=16)
plt.xlabel('Dead FMC', fontsize = 16)
plt.savefig('ResidualPlot.png')






