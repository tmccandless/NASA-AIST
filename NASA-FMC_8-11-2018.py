
# coding: utf-8

# In[1]:


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
import pydot
import matplotlib.pyplot as plt
import datetime
import h5py
import numpy.ma as ma


# In[2]:


# Load in HDF5 data
#store = pd.read_hdf('/glade/p/ral/wsap/NASA-FMC/sample_dfmc.2.h5','samples')
#storeAll.close()
storeAll = pd.HDFStore('/glade/p/ral/wsap/NASA-FMC/co_dfmc.h5')
store = storeAll.select('samples',where=('distNearestRaws < 2500','distNearestRaws > -9999.0'))
store = store.replace(-9999.0,np.NaN).dropna(axis=0,how='any')


# In[3]:


print(store['distNearestRaws'].shape)
#print(store.root[0][15].shape)


# In[4]:


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

print(predictorsI.shape)
print(predictandI.shape)
print(predictorsB.shape)
print(predictandB.shape)


# In[6]:


# Random forest method
# Split the data into training and testing sets
train_featuresI, test_featuresI, train_labelsI, test_labelsI = train_test_split(predictorsI, predictandI, 
                 test_size = 0.20, random_state = 42)
train_featuresB, test_featuresB, train_labelsB, test_labelsB = train_test_split(predictorsB, predictandB, test_size = 0.20, random_state = 42)
# Section below to understand the data after separating into training and testing datasets
print('Training Features Shape:', train_featuresI.shape)
print('Training Labels Shape:', train_labelsI.shape)
print('Testing Features Shape:', test_featuresI.shape)
print('Testing Labels Shape:', test_labelsI.shape)


# In[ ]:


######### Configure and train the Random Forest ##############
# Import the Random Forest Model
# Instantiate model 
rfI = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)
rfB = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)
# Train the model on training data
rfI.fit(train_featuresI, train_labelsI);
rfB.fit(train_featuresB, train_labelsB);
# Test other configurations of the Random Forest
#rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, 
                               #min_samples_split = 2, min_samples_leaf = 1)


# In[ ]:


# Use the forest's predict method on the test data
predictionsI = rfI.predict(test_featuresI)
predictionsB = rfB.predict(test_featuresB)

######### Compute Errors on Test Data #############
# Calculate the absolute errors
errorsI = abs(predictionsI - test_labelsI)
errorsB = abs(predictionsB - test_labelsB)
# Print out the mean absolute error (mae))
print('Mean Absolute Error with Indices for RF:', round(np.mean(errorsI), 2))
print('Mean Absolute Error with Reflectance Bands for RF:', round(np.mean(errorsB), 2))


# In[ ]:


# Feature selection
importancesI = list(rfI.feature_importances_)
importancesB = list(rfB.feature_importances_)
feature_listI = featuresI
feature_listB = featuresB
feature_importancesI = [(featuresI,round(importancesI,2)) for featuresI, importancesI in zip(feature_listI,importancesI)]
feature_importancesB = [(featuresB,round(importancesB,2)) for featuresB, importancesB in zip(feature_listB,importancesB)]
feature_importancesI = sorted(feature_importancesI,key = lambda x: x[1],reverse=True)
feature_importancesB = sorted(feature_importancesB,key = lambda x: x[1],reverse=True)
[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesI];
[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesB];


# In[ ]:


print(np.mean(test_labelsI))
print(np.mean(train_labelsI))
print(np.std(test_labelsI))
print(np.std(train_labelsI))
print(np.max(test_labelsI))
print(np.max(train_labelsI))
print(np.min(test_labelsI))
print(np.min(train_labelsI))


# In[7]:


# Test multiple linear regression
from sklearn.linear_model import LinearRegression
lmI = LinearRegression()
lmB = LinearRegression()
lmI.fit(train_featuresI,train_labelsI)
lmB.fit(train_featuresB,train_labelsB)
print("The coefficients for training on indices are the following: ", lmI.coef_)
print("The coefficients for training on reflectances are the following: ", lmB.coef_)
pred_lmI = lmI.predict(test_featuresI)
pred_lmB = lmB.predict(test_featuresB)
errors_lmI = abs(pred_lmI - test_labelsI)
errors_lmB = abs(pred_lmB - test_labelsB)
# Print out the mean absolute error (mae)
print('Mean Absolute Error with Indices for MLR:', round(np.mean(errors_lmI),2))
print('Mean Absolute Error with Reflectances for MLR:',round(np.mean(errors_lmB),2))


# In[14]:


plt.scatter(rfI.predict(train_featuresI),rfI.predict(train_featuresI)-train_labelsI,c='b',s=40,alpha=0.5)
plt.scatter(rfI.predict(test_featuresI),rfI.predict(test_featuresI)-test_labelsI,c='g',s=40)
plt.title('Residual Plot using training (blue) and test (green) data',fontsize=18)
plt.ylabel('Residuals',fontsize=16)
plt.xlabel('Dead FMC', fontsize = 16)


# In[15]:


# Let's try gradient boosted regression
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# Fit Regression Model
params = {'n_estimators':1000, 'max_depth': 6, 'min_samples_split':2, 'learning_rate': 0.03, 'loss':'ls'}
clfI = ensemble.GradientBoostingRegressor(**params)
clfB = ensemble.GradientBoostingRegressor(**params)
clfI.fit(train_featuresI,train_labelsI)
clfB.fit(train_featuresB,train_labelsB)
pred_clfI = clfI.predict(test_featuresI)
pred_clfB = clfB.predict(test_featuresB)
errors_clfI = abs(pred_clfI - test_labelsI)
errors_clfB = abs(pred_clfB - test_labelsB)
# Print out the MAE
print('Mean Absolute Error with Indices for GBR:',round(np.mean(errors_clfI),2))
print('Mean Absolute Error with Reflectances for GBR:',round(np.mean(errors_clfB),2))




# In[30]:


import tensorflow as tf
from tensorflow import keras

# First, normalize training data
meanI = train_featuresI.mean(axis=0)
meanB = train_featuresB.mean(axis=0)
stdI = train_featuresI.std(axis=0)
stdB = train_featuresB.std(axis=0)
trainI = (train_featuresI - meanI) / stdI
trainB = (train_featuresB - meanB) / stdB
testI = (test_featuresI - meanI) / stdI
testB = (test_featuresB - meanB) / stdB


def build_model(train_data):
    model = keras.Sequential([
        keras.layers.Dense(10, activation=tf.nn.relu,
                          input_shape=(train_data.shape[1],)),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    
    optimizer = tf.train.RMSPropOptimizer(0.001)
    
    model.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['mae'])
    return model

EPOCHS = 500
modelI = build_model(train_featuresI)
modelB = build_model(train_featuresB)
modelI.summary()

# Train model and store training stats
historyI = modelI.fit(trainI, train_labelsI, epochs=EPOCHS,
                      validation_split=0.2,verbose=0)
historyB = modelB.fit(trainB, train_labelsB, epochs=EPOCHS,
                      validation_split=0.2,verbose=0)

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [DFMC]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
            label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
            label='Val Loss')
    plt.legend()
    #plt.ylim([0,5])
    
plot_history(historyI)
plot_history(historyB)

[lossI, maeI] = modelI.evaluate(testI,test_labelsI,verbose=0)
print("Mean Absolute Error for Indices for tensorflow: ", maeI)
[lossB, maeB] = modelB.evaluate(testB,test_labelsB,verbose=0)
print("Mean Absolute Error for Reflectances for tensorflow: ", maeB)


# In[2]:


# Define plot and save figure

def plot_grid(grid_z,outfile):
    grid_z_pos = grid_z[np.where(grid_z != -9999.0)]
    print("min positive fuel moisture = %f, max positive fuel mositure = %f" % (min(grid_z_pos),np.max(grid_z_pos)))
    plt.matshow(grid_z,vmin=np.min(grid_z_pos)-1.0,vmax=np.max(grid_z_pos)+1.0,cmap='hot')
    plt.colorbar()
    if outfile:
        plt.savefig(outfile) 
    else:
        plt.show()


# In[5]:


#Height
#store.samples[0][-6]
#Width
#store.samples[0][-5]

#grid_z1 = store.root.samples[0][0]
#grid_z1 = grid_z1.reshape((683,872))
#plot_grid(grid_z1,None)

#grid_z2 = store.root.samples[1][0]
#grid_z2 = grid_z2.reshape((683,872))
#plot_grid(grid_z2,None)

#grid_z3 = store.root.samples[2][0]
#grid_z3 = grid_z3.reshape((683,872))
#plot_grid(grid_z3,None)

