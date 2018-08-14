{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/glade/u/home/mccandle/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n",
      "  from ._conv import register_converters as _register_converters\n"
     ]
    }
   ],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "File created by Tyler McCandless for the NASA-AIST Wildfire Project.\n",
    "\n",
    "\"\"\"\n",
    "# Imports Required\n",
    "# Pandas is used for data manipulation, h5py for data (conda install h5py)\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.tree import export_graphviz\n",
    "import pydot\n",
    "import matplotlib.pyplot as plt\n",
    "import datetime\n",
    "import h5py\n",
    "import numpy.ma as ma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Opening /glade/p/ral/wsap/NASA-FMC/co_dfmc.h5 in read-only mode\n"
     ]
    }
   ],
   "source": [
    "# Load in HDF5 data\n",
    "#store = pd.read_hdf('/glade/p/ral/wsap/NASA-FMC/sample_dfmc.2.h5','samples')\n",
    "#storeAll.close()\n",
    "storeAll = pd.HDFStore('/glade/p/ral/wsap/NASA-FMC/co_dfmc.h5')\n",
    "store = storeAll.select('samples',where=('distNearestRaws < 2500','distNearestRaws > -9999.0'))\n",
    "store = store.replace(-9999.0,np.NaN).dropna(axis=0,how='any')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(224707,)\n"
     ]
    }
   ],
   "source": [
    "print(store['distNearestRaws'].shape)\n",
    "#print(store.root[0][15].shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(224707, 6)\n",
      "(224707, 8)\n",
      "(224707, 5)\n",
      "(224707,)\n",
      "(224707, 7)\n",
      "(224707,)\n"
     ]
    }
   ],
   "source": [
    "# As example, set-up one random forest to train on\n",
    "dFMC = store['fuelMoisture']\n",
    "#index = store['index']\n",
    "distNearestRaws = store['distNearestRaws']\n",
    "ndvi = store['ndvi']\n",
    "ndwi = store['ndwi']\n",
    "gvmi = store['gvmi']\n",
    "pmi = store['pmi']\n",
    "vari = store['vari']\n",
    "b1 = store['one_km_Surface_Reflectance_Band_1']\n",
    "b2 = store['one_km_Surface_Reflectance_Band_2']\n",
    "b3 = store['one_km_Surface_Reflectance_Band_3']\n",
    "b4 = store['one_km_Surface_Reflectance_Band_4']\n",
    "b5 = store['one_km_Surface_Reflectance_Band_5']\n",
    "b6 = store['one_km_Surface_Reflectance_Band_6']\n",
    "b7 = store['one_km_Surface_Reflectance_Band_7']\n",
    "Ind = np.array([ndvi,ndwi,gvmi,pmi,vari,dFMC])\n",
    "Bands = np.array([b1,b2,b3,b4,b5,b6,b7,dFMC])\n",
    "featuresI = ['ndvi','ndwi','gvmi','pmi','vari']\n",
    "featuresB = ['b1','b2','b3','b4','b5','b6','b7']\n",
    "Ind = Ind.transpose()\n",
    "Bands = Bands.transpose()\n",
    "print(Ind.shape)\n",
    "print(Bands.shape)\n",
    "predictorsI = Ind[:,0:5]\n",
    "predictandI = Ind[:,5]\n",
    "predictorsB = Bands[:,0:7]\n",
    "predictandB = Bands[:,7]\n",
    "\n",
    "print(predictorsI.shape)\n",
    "print(predictandI.shape)\n",
    "print(predictorsB.shape)\n",
    "print(predictandB.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Features Shape: (179765, 5)\n",
      "Training Labels Shape: (179765,)\n",
      "Testing Features Shape: (44942, 5)\n",
      "Testing Labels Shape: (44942,)\n"
     ]
    }
   ],
   "source": [
    "# Random forest method\n",
    "# Split the data into training and testing sets\n",
    "train_featuresI, test_featuresI, train_labelsI, test_labelsI = train_test_split(predictorsI, predictandI, \n",
    "                 test_size = 0.20, random_state = 42)\n",
    "train_featuresB, test_featuresB, train_labelsB, test_labelsB = train_test_split(predictorsB, predictandB, test_size = 0.20, random_state = 42)\n",
    "# Section below to understand the data after separating into training and testing datasets\n",
    "print('Training Features Shape:', train_featuresI.shape)\n",
    "print('Training Labels Shape:', train_labelsI.shape)\n",
    "print('Testing Features Shape:', test_featuresI.shape)\n",
    "print('Testing Labels Shape:', test_labelsI.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "######### Configure and train the Random Forest ##############\n",
    "# Import the Random Forest Model\n",
    "# Instantiate model \n",
    "rfI = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)\n",
    "rfB = RandomForestRegressor(n_estimators= 500, random_state=2,criterion='mse',min_samples_split=4,min_samples_leaf=2)\n",
    "# Train the model on training data\n",
    "rfI.fit(train_featuresI, train_labelsI);\n",
    "rfB.fit(train_featuresB, train_labelsB);\n",
    "# Test other configurations of the Random Forest\n",
    "#rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, \n",
    "                               #min_samples_split = 2, min_samples_leaf = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use the forest's predict method on the test data\n",
    "predictionsI = rfI.predict(test_featuresI)\n",
    "predictionsB = rfB.predict(test_featuresB)\n",
    "\n",
    "######### Compute Errors on Test Data #############\n",
    "# Calculate the absolute errors\n",
    "errorsI = abs(predictionsI - test_labelsI)\n",
    "errorsB = abs(predictionsB - test_labelsB)\n",
    "# Print out the mean absolute error (mae)\n",
    "print('Mean Absolute Error with Indices:', round(np.mean(errorsI), 2))\n",
    "print('Mean Absolute Error with Reflectance Bands:', round(np.mean(errorsB), 2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Variable:pmi                  Importance: 0.23\n",
      "Variable:ndwi                 Importance: 0.21\n",
      "Variable:ndvi                 Importance: 0.2\n",
      "Variable:gvmi                 Importance: 0.19\n",
      "Variable:vari                 Importance: 0.17\n",
      "Variable:b7                   Importance: 0.21\n",
      "Variable:b2                   Importance: 0.16\n",
      "Variable:b3                   Importance: 0.15\n",
      "Variable:b6                   Importance: 0.13\n",
      "Variable:b5                   Importance: 0.12\n",
      "Variable:b1                   Importance: 0.11\n",
      "Variable:b4                   Importance: 0.11\n"
     ]
    }
   ],
   "source": [
    "# Feature selection\n",
    "importancesI = list(rfI.feature_importances_)\n",
    "importancesB = list(rfB.feature_importances_)\n",
    "feature_listI = featuresI\n",
    "feature_listB = featuresB\n",
    "feature_importancesI = [(featuresI,round(importancesI,2)) for featuresI, importancesI in zip(feature_listI,importancesI)]\n",
    "feature_importancesB = [(featuresB,round(importancesB,2)) for featuresB, importancesB in zip(feature_listB,importancesB)]\n",
    "feature_importancesI = sorted(feature_importancesI,key = lambda x: x[1],reverse=True)\n",
    "feature_importancesB = sorted(feature_importancesB,key = lambda x: x[1],reverse=True)\n",
    "[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesI];\n",
    "[print('Variable:{:20} Importance: {}'.format(*pair)) for pair in feature_importancesB];"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6.941458180669074\n",
      "6.963213935150535\n",
      "3.820447778195451\n",
      "3.679294031317135\n",
      "22.22301614509893\n",
      "22.33280252547348\n",
      "9.101568888228283e-06\n",
      "9.089005045725438e-06\n"
     ]
    }
   ],
   "source": [
    "print(np.mean(test_labelsI))\n",
    "print(np.mean(train_labelsI))\n",
    "print(np.std(test_labelsI))\n",
    "print(np.std(train_labelsI))\n",
    "print(np.max(test_labelsI))\n",
    "print(np.max(train_labelsI))\n",
    "print(np.min(test_labelsI))\n",
    "print(np.min(train_labelsI))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The coefficients for training on indices are the following:  [ -2.12294874 -10.52885979  10.06347259  -0.34648677   0.07648793]\n",
      "The coefficients for training on reflectances are the following:  [ 0.00376295 -0.00392776  0.00740153 -0.00739359  0.0054119   0.00047209\n",
      " -0.00517801]\n",
      "Mean Absolute Error with Indices: 2.3\n",
      "Mean Absolute Error with Reflectances: 2.21\n"
     ]
    }
   ],
   "source": [
    "# Test multiple linear regression\n",
    "from sklearn.linear_model import LinearRegression\n",
    "lmI = LinearRegression()\n",
    "lmB = LinearRegression()\n",
    "lmI.fit(train_featuresI,train_labelsI)\n",
    "lmB.fit(train_featuresB,train_labelsB)\n",
    "print(\"The coefficients for training on indices are the following: \", lmI.coef_)\n",
    "print(\"The coefficients for training on reflectances are the following: \", lmB.coef_)\n",
    "pred_lmI = lmI.predict(test_featuresI)\n",
    "pred_lmB = lmB.predict(test_featuresB)\n",
    "errors_lmI = abs(pred_lmI - test_labelsI)\n",
    "errors_lmB = abs(pred_lmB - test_labelsB)\n",
    "# Print out the mean absolute error (mae)\n",
    "print('Mean Absolute Error with Indices:', round(np.mean(errors_lmI),2))\n",
    "print('Mean Absolute Error with Reflectances:',round(np.mean(errors_lmB),2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5,0,'Dead FMC')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fff06d967b8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(rfI.predict(train_featuresI),rfI.predict(train_featuresI)-train_labelsI,c='b',s=40,alpha=0.5)\n",
    "plt.scatter(rfI.predict(test_featuresI),rfI.predict(test_featuresI)-test_labelsI,c='g',s=40)\n",
    "plt.title('Residual Plot using training (blue) and test (green) data',fontsize=18)\n",
    "plt.ylabel('Residuals',fontsize=16)\n",
    "plt.xlabel('Dead FMC', fontsize = 16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error with Indices: 2.08\n",
      "Mean Absolute Error with Reflectances: 1.88\n"
     ]
    }
   ],
   "source": [
    "# Let's try gradient boosted regression\n",
    "from sklearn import ensemble\n",
    "from sklearn.utils import shuffle\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# Fit Regression Model\n",
    "params = {'n_estimators':1000, 'max_depth': 6, 'min_samples_split':2, 'learning_rate': 0.03, 'loss':'ls'}\n",
    "clfI = ensemble.GradientBoostingRegressor(**params)\n",
    "clfB = ensemble.GradientBoostingRegressor(**params)\n",
    "clfI.fit(train_featuresI,train_labelsI)\n",
    "clfB.fit(train_featuresB,train_labelsB)\n",
    "pred_clfI = clfI.predict(test_featuresI)\n",
    "pred_clfB = clfB.predict(test_featuresB)\n",
    "errors_clfI = abs(pred_clfI - test_labelsI)\n",
    "errors_clfB = abs(pred_clfB - test_labelsB)\n",
    "# Print out the MAE\n",
    "print('Mean Absolute Error with Indices:',round(np.mean(errors_clfI),2))\n",
    "print('Mean Absolute Error with Reflectances:',round(np.mean(errors_clfB),2))\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_69 (Dense)             (None, 10)                60        \n",
      "_________________________________________________________________\n",
      "dense_70 (Dense)             (None, 1)                 11        \n",
      "_________________________________________________________________\n",
      "dense_71 (Dense)             (None, 1)                 2         \n",
      "=================================================================\n",
      "Total params: 73\n",
      "Trainable params: 73\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "Mean Absolute Error for Indices:  2.286041929244995\n",
      "Mean Absolute Error for Reflectances:  2.020002861658732\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXwAAAEKCAYAAAARnO4WAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJzt3Xl8VOW5wPHfM5N9Y0nCGiQJi8gSYowQJIiCWle07iCiBcW1aq22tre3FW/bq721Kl1UBKkL7huKVdwQRVkMyCIgskMEIQTZEkhI5r1/nJMQYDZC5kwy83w/n2PmnHnnnOeM4ck777yLGGNQSikV+VzhDkAppZQzNOErpVSU0ISvlFJRQhO+UkpFCU34SikVJTThK6VUlNCEr5RSUUITvlJKRQlN+EopFSViwh1AQxkZGSY7OzvcYSilVIuxcOHCHcaYzGDKNquEn52dTUlJSbjDUEqpFkNENgZbVpt0lFIqSmjCV0qpKKEJXymlokSzasNXSkWGgwcPUlpayoEDB8IdSsRISEggKyuL2NjYRp9DE75SqsmVlpaSmppKdnY2IhLucFo8Ywzl5eWUlpaSk5PT6PNok45SqskdOHCA9PR0TfZNRERIT08/7k9MmvCVUiGhyb5pNcX72fIT/sH98MVEWPdpuCNRSqlmreUnfHccfDkRSqaGOxKlVDNRXl5Ofn4++fn5dOjQgc6dO9fvV1dXB3WOn/3sZ6xatSroa06ePJm77rqrsSE7ouV/aetyQ68LYekrUF0JcUnhjkgpFWbp6eksXrwYgPvvv5+UlBTuueeew8oYYzDG4HJ5r/dOnRp5lciWX8MH6HsZHKyA5W+GOxKlVDO2Zs0a+vbty80330xBQQFbt25l/PjxFBYW0qdPHx544IH6ssXFxSxevJiamhpat27NfffdR//+/Rk0aBDbt28P+prPP/88/fr1o2/fvvz2t78FoKamhmuvvbb++MSJEwF45JFH6N27N/3792f06NFNe/NEQg0fILsYMk+CBU9C/ijQL4uUajYmvLOcFVv2NOk5e3dK4w8X9WnUa1esWMHUqVN54oknAHjwwQdp27YtNTU1nHnmmVx++eX07t37sNfs3r2boUOH8uCDD3L33Xfz9NNPc9999wW8VmlpKb/73e8oKSmhVatWnHXWWcyYMYPMzEx27NjBsmXLANi1axcAf/nLX9i4cSNxcXH1x5pSZNTwRWDAjbB1CWxeEO5olFLNWLdu3Tj11FPr91988UUKCgooKChg5cqVrFix4qjXJCYmct555wFwyimnsGHDhqCuNX/+fIYNG0ZGRgaxsbGMGjWKzz77jO7du7Nq1SruvPNOZs6cSatWrQDo06cPo0ePZtq0acc1wMqXyKjhA+RdBR9NsGr5JwwMdzRKKVtja+KhkpycXP949erVPPbYYyxYsIDWrVszevRor33d4+Li6h+73W5qamqCupYxxuvx9PR0li5dynvvvcfEiRN5/fXXmTRpEjNnzmT27NlMnz6dP/7xj3zzzTe43e5jvEPfIqOGDxCfAidfAyumw54t4Y5GKdUC7Nmzh9TUVNLS0ti6dSszZ85s0vMXFRUxa9YsysvLqamp4aWXXmLo0KGUlZVhjOGKK65gwoQJLFq0iNraWkpLSxk2bBj/93//R1lZGZWVlU0aT+TU8AEGjId5j8OCp+CsP4Q7GqVUM1dQUEDv3r3p27cvubm5DB48+LjON2XKFF577bX6/ZKSEh544AHOOOMMjDFcdNFFXHDBBSxatIhx48ZhjEFEeOihh6ipqWHUqFHs3bsXj8fDr3/9a1JTU4/3Fg8jvj5yhENhYaE57gVQXroGNn4Bv1ihXTSVCpOVK1dy0kknhTuMiOPtfRWRhcaYwmBeHzlNOnUG3Qb7f4QlL4Y7EqWUalZClvBF5EQRWdxg2yMioR+GdsIg6JhvNe14PCG/nFJKtRQhS/jGmFXGmHxjTD5wClAJhH5klIhVyy9fDWs+CvnllFKqpXCqSWc4sNYYE/Riu8el9yWQ2hHm/dORyymlVEvgVMK/GnCuUT0mzuqxs+5T+OEbxy6rlFLNWcgTvojEASOAV308P15ESkSkpKysrOkufMr1EJtkteUrpZRypIZ/HrDIGLPN25PGmEnGmEJjTGFmZmbTXTWpLfQfCctegX3BT3SklGr5zjjjjKMGUT366KPceuutfl+XkpJyTMdbGicS/kicbM5pqOgWqK2Gr6aE5fJKqfAYOXIkL7300mHHXnrpJUaOHBmmiJqHkCZ8EUkCzgbeCOV1fMroAT1+Al9NhoPHtxakUqrluPzyy5kxYwZVVVUAbNiwgS1btlBcXMy+ffsYPnw4BQUF9OvXj+nTpzfqGhs3bmT48OHk5eUxfPhwNm3aBMCrr75K37596d+/P6effjoAy5cvZ8CAAeTn55OXl8fq1aub5kaPUUinVjDGVALpobzG/upabnp+Ief0bs/ooq5HFxh0Kzx7MSx7FQquDWUoSilv3rsPfljWtOfs0A/Oe9Dn0+np6QwYMID333+fiy++mJdeeomrrroKESEhIYE333yTtLQ0duzYQVFRESNGjDjmNWNvv/12xowZw3XXXcfTTz/NHXfcwVtvvcUDDzzAzJkz6dy5c/0Ux0888QR33nkn11xzDdXV1dTW1h7X7TdWix9pmxjnZtvuA7y9xMeEaTlDoX1f68vbZjSNhFIqtBo26zRszjHG8Nvf/pa8vDzOOussvv/+e7Zt8/oVo19z585l1KhRAFx77bXMmTMHgMGDB3P99dfz1FNP1Sf2QYMG8ec//5mHHnqIjRs3kpiY2BS3eMwiYvK08/p14LGPV7N9zwHapSUc/qSI1ZY//Tarm2a3M8MSo1JRy09NPJQuueQS7r77bhYtWsT+/fspKCgAYNq0aZSVlbFw4UJiY2PJzs72OiXysar7hPDEE08wf/583n33XfLz81m8eDGjRo1i4MCBvPvuu/zkJz9h8uTJDBs27LiveaxafA0f4IJ+HTEGZi7/wXuBvpdDcibM+5ezgSmlwiYlJYUzzjiDsWPHHvZl7e7du2nXrh2xsbHMmjWLjRsbNx70tNNOq/8EMW3aNIqLiwFYu3YtAwcO5IEHHiAjI4PNmzezbt06cnNzueOOOxgxYgRLly49/htshIhI+D3ap9K9XQrTF/to1olNgFNvgNUfQNl3zganlAqbkSNHsmTJEq6++ur6Y9dccw0lJSUUFhYybdo0evXqFfA8lZWVZGVl1W9/+9vfmDhxIlOnTiUvL4/nnnuOxx57DIB77723fq3a008/nf79+/Pyyy/Tt29f8vPz+fbbbxkzZkzI7tmfiJkeefLn6/jjuyuZ8fNi+nZudXSBfWXwSB9rkZQLHznOSJVS/uj0yKGh0yPbrijsQlKcm6e/WO+9QEom5F0Ji1+Eyp3OBqeUUs1AxCT8VomxXHFKFu8s2cL2vT6+gCm6FWr2w8KpzganlFLNQMQkfIDrB+dwsNYwbd4m7wXa94bcM2H+JKipdjY4paJMc2oujgRN8X5GVMLPyUhmWK92TJu/kQMHfQxsGHQb7PsBlod+an6lolVCQgLl5eWa9JuIMYby8nISEhICF/YjIvrhNzR2cA6jp8znnSVbuKKwy9EFug2HjJ7WXPl5V1r99JVSTSorK4vS0lKadAbcKJeQkEBWVtZxnSPiEv7g7un0bJ/C019s4PJTso4eLu1yWQOxZvwCNn4J2ce3Sr1S6mixsbHk5OSEOwx1hIhq0gFrtNvYwTms3LqHuevKvRfKuxoS28JcXRFLKRU9Ii7hA1xycmfSk+OY8rmPLppxSVA4Flb9B8rXOhucUkqFSUQm/IRYN6OLuvLxt9tZW7bPe6EBN4IrBuY/6WxwSikVJhGZ8AFGF3UlLsbF03N81PJTO0C/y+Hr52H/LmeDU0qpMIjYhJ+ZGs9P8zvz+qJSdlb46HNfdCscrIBFzzgbnFJKhUHEJnyAcUNyOHDQwwvzfcyG1zEPsodYA7Fqa5wNTimlHBbRCb9n+1SG9MjgmbkbqarxMRCr6FbYUworG7fMmVJKtRQRnfABbhiSS9neKmYs2eq9QM9zoW2u1UVTRwUqpSJYxCf803tk0KNdCpPnrPc+zNvlsmr53y+EzQucD1AppRwS8QlfRLhhiD0Qa62PgVj9R0JCK2u6BaWUilARn/ABLs63BmJN9tVFMz4FTrkeVr4DPzZuuTOllGruoiLhJ8S6uXZQVz75djtrtvsaiHUTiEsHYimlIlZUJHxoMBDL14pYrTpD70tg0bNwYI+zwSmllAP8zpYpIm8HcY6dxpjrmyac0MlIsQZivbGolHvOOZG2yXFHFxp0K3zzmjX6dtCtzgeplFIhFGh65JOAG/w8L0CL+aZz3JAcXi7ZzAvzN3L7sB5HF+h8CpwwCOY/DgNvApfb+SCVUipEAiX8/zLGzPZXQEQmNGE8IdWzfSqn98zkmbkbufH0XOJjvCT0olvhlWvh2xnQ+2Lng1RKqRAJ1Ib/tohkHnlQRNqJSAKAMeaVkEQWIjcU51C2t4p3fA3E6nUBtO4Kc//lbGBKKRVigRL+RGCIl+NnA480fTihN6RHBj3bpzD583U+BmK5YeDNsHkelC50PkCllAqRQAm/2BjzxpEHjTHTgNMDnVxEWovIayLyrYisFJFBjQ20qYgINxTn8u0Pe/nS10Csk0dDfJoOxFJKRZRACd/fCt/BdOl8DHjfGNML6A+sDDawUBqR34mMlDim+BqIlZAGBWNg+Vuwu9TZ4JRSKkQCJe3tIjLgyIMicirgdzl6EUnD+hQwBcAYU22MaRYrjdStiOV/INZ4wMCCSY7GppRSoRIo4d8LvCIi94vIRfY2AXjFfs6fXKw/ClNF5GsRmSwiyUcWEpHxIlIiIiVlZX7/hjSpgAOx2nSFky6Ckn9DlY8/Ckop1YL4TfjGmAXAQKymnevtTYCBxpj5Ac4dAxQAjxtjTgYqgPu8XGOSMabQGFOYmXlUh6CQyUiJ59KTO/P6Qj8rYg26Hap2w+IXHItLKaVCJWA7vDFmmzHmD8aYy+zt98aY7UGcuxQobfCH4TWsPwDNxtjiHKpqPEyb52PCtC4DoHOhNRDL43E2OKWUamJ+E76ILPWxLRORpf5ea4z5AdgsIifah4YDK5oo7ibRs30qQ+2BWD5XxBp0K+xcB9+952xwSinVxALV8D1ALfAccCVwkb1daP8M5OfANPuPQz7w58aHGho3DMlhx74q3l68xXuBky6GVl2sFbGUUqoFC9SGnw+MBFKAF4A/AX2A740xASeON8Ysttvn84wxlxhjfmyKoJtScfcMTmyfyhRfK2K5Y6yBWBu/gO8XOR+gUko1kWDa8L+12/ALgHeAZ4FfhDwyh4gI44pz/A/EKhhjDcSa+w9ng1NKqSYUMOGLSGcR+aWIzAFGYyX7x0MemYPqBmJN/nyd9wINB2Lt2uxscEop1UQCfWk7G6tWH4vVJfM64F0gTkTahjw6hyTEurm2KJtZq8pYs32v90IDb7Z+zn/CucCUUqoJBarhdwXaADcBHwAl9rbQ/hkxRhedQFyMiylzNngv0LoL9LkEFj6jK2IppVqkQF/aZhtjcuwtt8GWY4zJdSpIJ6SnxHNZgbUilt+BWNV7rWUQlVKqhQnUpHN7g8d9Qh9OeI0dHGAgVucC6DrYataprXE2OKWUOk6BmnTGNnj8XCgDaQ56tE/ljBMDDcS6HXZvhhVvORucUkodp2CmOK7jb6rkiDGuOMBArJ7nQttuVhdNb/32lVKqmQqU8FuLyE9F5DIgTUQubbg5EaDTAg7Ecrms6Ra2fA2b5jofoFJKNVKghD8bGIE1lcJnHJpaoW56hYgjIowbYg3E+mKNj4FY/UdBYlv4UgdiKaVajhh/TxpjfuZUIM3Jxfmd+Mv7q5g8Zx3FPTKOLhCXBKeOg8/+CuVrIb2b80EqpdQxCmak7Yki8rCIvGtvfxWRnk4EFy7xMW7GDOrKp/4GYp16I7hjdVI1pVSLEahb5iDgU2AfMAl4Cmshk09FpCjk0YXRNQNPIN7fQKzU9pB3pbU4SuVOR2NTSqnGCFTD/z0w0p48bbox5i1jzB+wZtD8Q+jDC5/0lHguLcjijUWllO+r8l6o6Dao2Q8lU5wNTimlGiFQwu9mjPn0yIPGmNlYa9ZGtHHF2dZArPmbvBdo3xu6DYcFT0GNjz8KSinVTARK+D4asAGraSeidW9nDcR6du4GDhz0MRDrtNth3zZY9qqjsSml1LHy20sH6CIiE70cF6BzCOJpdm4ozmX0lPm8vWQLVxZ2ObpA7pnQvq/15W3+NSBRMT5NKdUCBarh34s1M+aRWwnwq9CG1jwM7p5Orw6pPO1rIJYIDLoNtq+AtZ84H6BSSgUpUD/8Z5wKpLmqWxHr3teW8sWacu/98vteBh/db0230H244zEqpVQwAnXLvD/QCYIp09JZK2LFM3mOjxWxYuJhwHirhr9tubPBKaVUkAK14d8gIv5W+xDgauD+JouoGaobiPW3D79j9ba99GifenShwrHw+cNWW/4l/3I+SKWUCiBQG/5TQKqfLcUuE/HqBmI9/cV67wWS2lpf2i59Bfb+4GxwSikVhEBt+BOcCqS5qxuI9fqiUu4550TSU+KPLlR0C3w12eqXP/y/nQ9SKaX8OJb58KPeuOJsqms8PD/Px0Cs9G7Q6wJr5G3VPmeDU0qpADThH4Pu7VI588RMnpvnZyDW4Dth/4/wdcQvEKaUamGCmS3TLSK/cCKYluCGIbns2FfN20t8rIjVZQCccJr15W3tQWeDU0opPwImfGNMLXCxA7G0CKd1swZiTfncx0AsgOK7rHVvv3nd2eCUUsqPYJt0vhCRf4jIEBEpqNtCGlkzVTcQa9W2vcxZs8N7oR7nQLve8MVjuu6tUqrZCDbhnwb0AR4AHra3vwZ6kYhsEJFlIrJYREoaH2bzUj8Q63MfXTRF4LQ7rOkWVn/gbHBKKeVDUAnfGHOml21YkNc40xiTb4wpPI44m5X4GDfXDerK7O/KWL3Nx4Si/S6HtCyY86izwSmllA9BJXwRaSUifxOREnt7WERahTq45uyaoq7+B2K5Y61J1TZ9CZsXOBucUkp5EWyTztNYc+NfaW97gKlBvM4AH4jIQhEZ37gQm6e2yXFcdkoWry/63veKWAVjIKG11ZavlFJhFmzC72Yvc7jO3iYQ3IpXg40xBcB5wG0icvqRBURkfN0nh7KysmMIPfzGDs7xPxArPsWaVO3bd6HsO2eDU0qpIwSb8PeLSHHdjogMBvYHepExZov9czvwJjDAS5lJxphCY0xhZmZmkOE0D93bpTCsVzv/A7EG3gQxCfCl1vKVUuEVbMK/Gfin3etmA/AP4CZ/LxCRZBFJrXsMnAN8cxyxNkvjinOsgViLfQzESs6Ak0fDkpdhj48ySinlgGBG2rqAE40x/YE8IM8Yc7IxZmmAl7YH5ojIEmAB8K4x5v3jjriZqRuINXnOOt8DsU67HYwH5um0yUqp8AlmpK0HuN1+vMcY429+/IavW2eM6W9vfYwxfzrOWJslEeGGIbl8t20fn6/2MRCrTTb0+SmU/Bv273IyPKWUqhdsk86HInKPiHQRkbZ1W0gja0Eu6t+RdqnxTPrMx4pYAIPvgOq91kyaSikVBsEm/LHAbcBnHL6QucIaiDW2OIc5a3bwzfe7vRfq2B+6DYN5T8DBA84GqJRSBN+GP9oYk3PEFky3zKgxauAJpMTH8MTstb4LDb4LKrbDkhecC0wppWzBtuEHnDcn2qUlxHLNwBP4z7KtbCqv9F4o53TodDJ8+Xfw+OjGqZRSIRJsk84HInKZiEhIo2nhxhbn4HYJT33uoy1fxKrl71wHK99xNjilVNQLNuHfDbwKVInIHhHZKyJB9daJJu3TEvjpyZ15pWSz7+kWTroI2naDLx7VqZOVUo4KdrbMVGOMyxgTZ4xJs/fTQh1cSzT+9G5U1Xh4Zu5G7wVcbjjt57Dla1j/mbPBKaWimt+ELyKjGzwefMRzt4cqqJase7sUzu7dnmfnbqCyusZ7of4jIbmdVctXSimHBKrh393g8d+PeG5sE8cSMW4e2o1dlQd5+avN3gvEJkDRLbD2E9i6xNnglFJRK1DCFx+Pve0r2yld23Bqdhsmf76eg7Ue74UKx0Jcqk6drJRyTKCEb3w89ravGrh5aDe+37Wfd5du9V4gsTUUXg/L34SdPhZRUUqpJhQo4fcSkaUisqzB47r9Ex2Ir8U688R29GiXwhOz1/qeVK3oVhA3zP2Hs8EppaJSoIR/EnARcGGDx3X7vUMbWsvmcgk3De3Gtz/sZfZ3PhZ2SesE/a+Cr5+HfS1r8RelVMvjN+EbYzb625wKsqUa0b8THdIS/E+3cNqdUFMFCyY5F5hSKioFO/BKNUJcjItxxTnMW7eTxZt9TIuc2RN6XWAl/Kp9zgaolIoqmvBDbOTAE0hNiOHJQJOqHdgFi551LjClVNQ55oQvIm1EJC8UwUSilPgYri3qyvvLf2D9jgrvhbqcCl0Hw9x/Qu1BZwNUSkWNoBK+iHwqImn2oidLgKki8rfQhhY5rh+cTazb5XtSNbBq+XtKYdlrzgWmlIoqwdbwW9lLG14KTDXGnAKcFbqwIku71AQuK8jitYWlbN/rY/GTHmdDuz7WQCyPj8FaSil1HIJN+DEi0hG4EpgRwngi1vjTczlY6+GZLzd4LyACg++EspWw+gNHY1NKRYdgE/4DwExgrTHmKxHJBVaHLqzIk5ORzLl9OvDc3I3sq/IxqVrfS6FVF51UTSkVEsFOj/yqMSbPGHOLvb/OGHNZaEOLPDcP7caeAzW8tGCT9wLuWBh0G2yaC5vmOxucUiriBfulba6IvCMiZSKyXUSmi0hOqIOLNP27tKYoty2TP19PdY2PdvqCMZDYRmv5SqkmF2yTzgvAK0BHoBPW6lcvhSqoSHbz0G78sOcAby/Z4r1AXDIMGA+r/gNlq5wNTikV0YJN+GKMec4YU2Nvz6OzZTbK0J6Z9OqQypOz1+Lx+HgLB9wEMYnwxURng1NKRbRAK161tfvezxKR+0QkW0S6isivgHedCTGyiAg3Dc1l9fZ9zFq13Xuh5HQouBaWvgy7v3c2QKVUxApUw18IlABXATcBs4BPgVuAn4U0sgh2YV4nOrdO9D+p2qDbwXhg3r+cC0wpFdECzZaZY4zJtX8etqHz4TdarNuaVO2rDT+ycONO74XadLW6aS78N+z/0dH4lFKR6Zjm0hHLMBGZDJSGKKaocPWALrROiuXJ2f6mW7gTqvfBV1OcC0wpFbGC7ZY5UEQeAzYCbwOfA72CfK1bRL4WER2h20BSXAxjirry4cptrNnuY1rkDv2g23CY/wQc3O9sgEqpiBPoS9s/ichq4M/AMuBkoMwY84wxJth2hjuBlccXZmS67rRs4twunvrMTy2/+C6oKIPFLzgXmFIqIgWq4Y8HtgGPA88bY8o5hu6YIpIFXABMbnSEESw9JZ4rC7vw5tffs22Pj0nVsodApwL48u/gqXU2QKVURAmU8DsAfwJGAGtE5DkgUURigjz/o8CvAJ/TP4rIeBEpEZGSsrLoW9f1xiG51Hg8PP3Feu8FRKxa/o/rYcV0Z4NTSkWUQL10ao0x7xljxgDdgenAl8D3IuK3jUFELgS2G2MWBrjGJGNMoTGmMDMz8xjDb/lOSE/i/H4deWHeJvYc8LH4Sa8LoW03a+pko+PdlFKNE3QvHWPMAWPMa/akaT2wZs/0ZzAwQkQ2YE3DMExEnm90pBHs5qHd2FtVwwvzfUyq5nLD4Dtg62JYP9vZ4JRSEaNRa9oaY/YYY54JUOY3xpgsY0w2cDXwiTFmdGOuF+n6dm5FcfcMnp6znqoaH+30eVdDSnuYo5OqKaUaRxcxbyZuGprL9r1VvPW1j6kUYhOg6BZYNwtKS5wNTikVERxJ+MaYT40xFzpxrZaquHsGfTql8eRn63xPqnbqDZCUDp/80dnglFIRIeiELyKnicgoERlTt4UysGhjTarWjXVlFXy4cpv3QvGpUPwLq5a/YY6zASqlWrxgR9o+B/wVKAZOtbfCEMYVlc7v24Euba1J1Yyv3jiF4yClA3zyJ+2xo5Q6JsHW8AuBwcaYW40xP7e3O0IZWDSKcbu4cUguX2/axVcbfAxkjkuC0++BTV/C2o+dDVAp1aIFm/C/wRqEpULsilO60DY5jif9TZ1cMAZanQAfTQCPzzFtSil1mGATfgawQkRmisjbdVsoA4tWiXFurhuUzcffbmfVD3u9F4qJh+G/hx+WwrJXnA1QKdViBZvw7wcuwZpE7eEGmwqBMYO6khjrZpK/SdX6XgYd8+Hj/9GZNJVSQQkq4RtjZnvbQh1ctGqTHMdVp3Zh+uLv2bLLRzJ3ueCcP8KeUpj3uLMBKqVapGB76RSJyFcisk9EqkWkVkT2hDq4aHbDkBwM8PQcH5OqAeQMgZ7nwZxHoGKHY7EppVqmYJt0/gGMBFYDicAN9jEVIlltkrgoryMvLtjE7kofk6oBnD0Bqitg9l+cC04p1SIdy+RpawC3PYPmVOCMkEWlABh/ejcqqmt5fv5G34UyT7R67ZRMgXI/PXuUUlEv2IRfKSJxwGIR+YuI/AJIDmFcCujdKY2hPTOZ+sV6Dhz0s/jJGb8Bdzx8dL9jsSmlWp5gE/61dtnbgQqgC3BZqIJSh9w0NJcd+6p5fZGfNeNT21uLpKx8GzbNcy44pVSLEmwvnY2AAB2NMROMMXfbTTwqxAblptM/qxVPfbaOWl+TqgEMug1SO8H79+lgLKWUV8H20rkIWAy8b+/n68ArZ9RNqrahvJJ3l231XTAu2foCd8vXsORF5wJUSrUYxzLwagCwC8AYsxjIDk1I6kg/6dOBHu1SmPjxav+1/H5XQNapVlv+Ae01q5Q6XLAJv8YYszukkSif3C7hzrN6sGb7PmYs3eK7oAic+xBUbIfPdSC0UupwQU+eJiKjALeI9BCRv2MtZq4ccn7fjpzYPjVwLT/rFOg/Cub9S7tpKqUOE2zC/znQB6gCXgT2AHeFKih1NJddy19bVuG/lg9w1h/AHQfO5aUvAAASx0lEQVTv/VrnzFdK1Qu2l06lMea/jDGnGmMK7ccHQh2cOty5fTrQq0MqjwWq5ad2gDN/C2s+tLpqKqUUARJ+w6mQvW1OBaksLpdw11k9WFdWwdtLfCx2XmfATdC+H7x3H1T5mGZZKRVVYgI8PwjYjNWMMx+rL74Ko3N6d+Ckjmk89tFqLszrRKzbx99sdwxc+AhMORtm/S+c+2dnA1VKNTuBmnQ6AL8F+gKPAWcDO3R65PBxuYR7zunJhvJKXlywyX/hLqfCKdfD/Cdg61JH4lNKNV9+E749Udr7xpjrgCJgDfCpiPzckeiUV8N6taMoty2PfbSavQf8zKQJ1he4iW3g3bt1BK5SUS7gl7YiEi8ilwLPA7cBE4E3Qh2Y8k1E+K/ze1NeUc0T/ta+BSvZ/+RPUPoVLHrGmQCVUs1SoC9tn8Hqb18ATLB76fyPMSbAN4Yq1PplteLi/E5M/nw9W3cHWOIw7yrIHgIf/QH2lTkToFKq2QlUw78W6AncCXwpInvsba+ueBV+95xzIsbAwx9857+gCFzwN6iuhA//25nglFLNTqA2fJcxJtXe0hpsqcaYNKeCVN51aZvEdad15fVFpazYEuDvb2ZPGHynNbHa+s+dCVAp1awEveKVap5uP7MHaQmx/O97KwMXPv0eaN0V3v0l1FSHPjilVLMSsoQvIgkiskBElojIchGZEKprRbNWSbH8fFh3Pl+9g1nfbvdfODYRzv8r7FgFc//uTIBKqWYjlDX8KmCYMaY/kA+cKyJFIbxe1BozKJvczGTuf2e5/6UQAXqeAyeNsBY9/3GDI/EppZqHkCV8Y9ln78bam87kFQJxMS4eGNGXjeWVTPpsXeAXnPsguGLgP/fq5GpKRZGQtuGLiFtEFgPbgQ+NMfNDeb1oVtwjgwvyOvLPWWvYvLPSf+FWna3J1VZ/ACvfcSZApVTYhTTh2yN184EsYICI9D2yjIiMF5ESESkpK9M+4sfjdxechNslTHhnReDCdZOrva+TqykVLRzppWOM2QV8Cpzr5blJ9pTLhZmZmU6EE7E6tkrkjuE9+GjlNj5euc1/YXcMXPQo7N1qzaiplIp4oeylkykire3HicBZwLehup6yjB2cQ7dgv8DNKoQhv4TFz8PSV5wJUCkVNqGs4XcEZonIUuArrDb8GSG8nsL+AvfivmzeuT/wPDsAQ38NJ5wG02+HzQtCH6BSKmxC2UtnqTHmZGNMnjGmrzHmgVBdSx1ucPcMLszryL8+Xcum8gBf4Lpj4arnIa0TvDhSu2oqFcF0pG2E+t0FvYlxCb+b/g0mUNfL5HS45lXwHIQXroIDu50JUinlKE34EapDqwR+fW4vPvuujJe/2hz4BRk94MrnoHwNvHId1AaYZ18p1eJowo9g1xZ1ZVBuOv8zY0XgvvkAuUOtZRHXzbLm29FBWUpFFE34EczlEv5yeR4Av3ptKR5PEAm8YAwU320tljL7oRBHqJRykib8CNelbRL/fWFv5q4r59m5G4J70fDfQ/9R8On/wldTQhmeUspBmvCjwFWnduGMEzN58P1vWb+jIvALRGDEROhxDvznHp1+QakIoQk/CogID16aR5zbxT2vLqE2mKYddyxc8W/ofAq8Ng42fBHyOJVSoaUJP0p0aJXAhIv7sHDjj0z+PIgZNQHikmHUK9Cmq9VHf9vy0AaplAopTfhR5JL8zpzTuz0Pf/Ad320LcsK0pLYw+g0r+T93KZQFWD9XKdVsacKPIiLCn37aj5SEGH75yhIO1nqCe2HrLjD6dTC1MOVs2Dg3tIEqpUJCE36UyUyN54+X9GXZ97t5/NMg5tqp0743jPsQktLh2Yt1sjWlWiBN+FHo/H4dGdG/ExM/Xs3yLccwjULbHCvpdy6AN26EV6+HivKQxamUalqa8KPUAxf3oU1yHL98ZQlVNQGmUW4oOR2um2H11V85A/5VBKveC12gSqkmowk/SrVOiuPBS/vx7Q97efSj1cf2YneMNY/++FmQ0g5evBqevxzWzgLPMfzxUEo5ShN+FBt+UnuuKuzC45+u5b1lW4/9BB36wY2fwFn3Q+lX8Nwl8HAveOs2WPaaNvco1cxIwKlzHVRYWGhKSkrCHUZUOXCwlpFPzWPl1j08O3YgA3LaNu5EB/fDdzNhxVtWTf/ALkCgY3/odqY1gKtjf2jVxRrJ6xRPLdRUgafm0FZ78PD9umOm1powzhgwnqM36o7bZQ7br3ve/vckYt2/CIjr0GMEBOz/+BGCf5emwXnr/90HsX8sZev3j3wO72XryeG/Fw3fv8OeP4Zjh9944LiDvrdgz9uwaIDzxsRDn58e/bogiMhCY0xhUGU14asd+6q48sm5bN9TxfM3DCS/S+vjO6GnFrYshrWfwNqPrZW0jN3Uk9gGOuRB+76Qkmn1+klKh7gUa3SvKwZcbqiphoMV1h+S6ko4WAnV+6y5+g/shv27Dj2u2W8l9ZoD9s+qQ/tGm5hUC5DcDu49xqZVmyZ8dcx+2H2AK5+cy67Kal4cX0SfTq2a7uTVlbB9BWxdcmgrW2Ul6mMmkNDK2hJbQ3yaNSjMHQcxCVZNqf6n/dgdCy77j4k7xv6j0uCPS90fGnFbtXFxHaqZH/kYOfxYfS2+wX59rfiIn0ceC1TLD8knoSNqww1+HNo/8nkJ8JyP/WDLHvWJwBznMTjq/Q3qHoMtc+Qnkcaet8E5xG2NaG8ETfiqUTbvrOTKJ+dSVePh5fFF9GifGtoLVldCZTlU7rAee+qaWmqtBB6bBLGJVkKPTbT249PApV89KVVHE75qtPU7KrjySWsk7Ys3FtG9XUqYI1JK+XMsCV+rSuowORnJvHDDQIwxjHxqHmvL9oU7JKVUE9GEr47So30qL95YZCX9SfNYp0lfqYigCV951aN9Ki/cWITHGK58ci5LNu8Kd0hKqeOkbfjKr7Vl+7h+6gJ27K1m4siTObt3+3CH1CSMMdR4DFU1Hiqra9hfXUt8jJtaY4h1CdW1HqprPPU/D9Z6qPVAjcc+XuOhxmPwGINLBJdYs5HWPXaJ1d++/jmkvnOGUFcW3C457LFLBI8x9ddwuYRYtwswVNcYajye+i7cpsG9NNw/rFs5Bo8HPMYQ4xZiXC6qazwcqKmlptbgdgmxbsHtcmGMdT+HojRW5yLsTkaH7dvXNNa5Gx6v6yRj/LweY/CYQ+WPPK+IkJoQg0vE5zldYo0Yj49xIfabW1+2wftQ9xoaXOfI9800eNHR7691Dl+vp8HrDyvX4Dze/h81LJsQ6+aSkzvTGMfShh/TqCuoqNEtM4U3bhnMuGe+4qbnSrj9zO7cMbwHMe7wfzisqqllZ0U1Oyuq+bHiIDsrq/mxbr+ymvKKanZVVrPvQA17q2qoqKqhsrqWKjuBN6O6jopyGSnxjU74x0ITvgooMzWel8YX8d9vLWfiJ2v4bPUO/vzTfvTulBbya++vrmVDeQUbdlSwobySDTsqWF9ewcbyCrbtqfL5utZJsbRNiqNNchytk+LIaptESlwMiXFu4mNdxLtdxMVYW1JcDElxbiqqa4l1CTUeQ1yD5+PcLmJjXLhFcLuEuBgX8TEuYtxi10CtWm5dTddj116t/bqarBVXwxpt3Ws8BjyeQ48FiI91Eet24fEYDtpLUsa5667Z8E6tnYafHqz9Q4XcYn26qPEYamo9xMdY74HbJdR6DAdrPXg81jlcdTVlTP2nEmnwCUXqr3XoU03dscM+xfh4ru7TzWGPG5Sp677v8Rj2HqjBGHye02Pgx4pqqms9dg360Htx5PtwKO7D46x/F+uuX3+dQ2WPPqf3973htbydk8Nef+hewPp05wRt0lHHZMbSLfxh+nJ+rKxm1MATuP3MHnRolXBc59xfXcvGnVZSX7+jko3lFazfUcEGL0k9IyWO7PRksjOSOaFtEhkp8bRNjqVNUhzpKXG0SYqjVWJss/gEopQTtB++CqndlQd5+MNVvDB/Ey4Rrjw1i8sKsuif1RrXETWVupraj5XVlO2rYtueA/yw+wAb7KS+rqyCrbsPHPaajJQ4uqYnk52eTE5GEl3Tk8nJSKZrehKpCbFO3qpSzZ4mfOWIzTsr+dena3lt4WYO1hpS42NIS4wlxi1UVteyv7qWyuoaPF5+xVITYsjNTKFbhp3MM5LJSU+ma0YSaZrUlQpas0j4ItIFeBboAHiAScaYx/y9RhN+y7SrsprZ35VRsuFHKqprqPUYkuLcJMbGkBLvplVSHK0TY8lMjad9WgLt0+JplRh7WDuzUqpxmksvnRrgl8aYRSKSCiwUkQ+NMStCeE0VBq2T4rg4vzMX54e+l4FSqvFC9s2WMWarMWaR/XgvsBLQjKCUUmHiSFcGEckGTgbmO3E9pZRSRwt5wheRFOB14C5jzB4vz48XkRIRKSkrKwt1OEopFbVCmvBFJBYr2U8zxrzhrYwxZpIxptAYU5iZmRnKcJRSKqqFLOGL1QVjCrDSGPO3UF1HKaVUcEJZwx8MXAsME5HF9nZ+CK+nlFLKj5B1yzTGzCHgop1KKaWcohOOKKVUlGhWUyuISBmwsZEvzwB2NGE4LYHec3TQe44Ojb3nrsaYoHq8NKuEfzxEpCTY4cWRQu85Oug9Rwcn7lmbdJRSKkpowldKqSgRSQl/UrgDCAO95+ig9xwdQn7PEdOGr5RSyr9IquErpZTyo8UnfBE5V0RWicgaEbkv3PE0FRF5WkS2i8g3DY61FZEPRWS1/bONfVxEZKL9HiwVkYLwRd54ItJFRGaJyEoRWS4id9rHI/a+RSRBRBaIyBL7nifYx3NEZL59zy+LSJx9PN7eX2M/nx3O+I+HiLhF5GsRmWHvR/Q9i8gGEVlmzzpQYh9z9He7RSd8EXED/wTOA3oDI0Wkd3ijajL/Bs494th9wMfGmB7Ax/Y+WPffw97GA487FGNTq1s05ySgCLjN/v8ZyfddBQwzxvQH8oFzRaQIeAh4xL7nH4FxdvlxwI/GmO7AI3a5lupOrHUy6kTDPZ9pjMlv0P3S2d9tY0yL3YBBwMwG+78BfhPuuJrw/rKBbxrsrwI62o87Aqvsx08CI72Va8kbMB04O1ruG0gCFgEDsQbgxNjH63/PgZnAIPtxjF1Owh17I+41CyvBDQNmYE3DEun3vAHIOOKYo7/bLbqGj7WC1uYG+6VE9qpa7Y0xW8FaUQxoZx+PuPfhiEVzIvq+7aaNxcB24ENgLbDLGFNjF2l4X/X3bD+/G0h3NuIm8SjwK6z1rsG6h0i/ZwN8ICILRWS8fczR3+1QrmnrBG+Ts0Vjt6OIeh+OXDTHz2LnEXHfxphaIF9EWgNvAid5K2b/bPH3LCIXAtuNMQtF5Iy6w16KRsw92wYbY7aISDvgQxH51k/ZkNxzS6/hlwJdGuxnAVvCFIsTtolIRwD753b7eMS8Dz4WzYn4+wYwxuwCPsX6/qK1iNRVyBreV/0928+3AnY6G+lxGwyMEJENwEtYzTqPEtn3jDFmi/1zO9Yf9gE4/Lvd0hP+V0AP+9v9OOBq4O0wxxRKbwPX2Y+vw2rjrjs+xv5mvwjYXfcxsSUR8bloTsTet4hk2jV7RCQROAvri8xZwOV2sSPvue69uBz4xNiNvC2FMeY3xpgsY0w21r/ZT4wx1xDB9ywiySKSWvcYOAf4Bqd/t8P9RUYTfBFyPvAdVrvnf4U7nia8rxeBrcBBrL/247DaLT8GVts/29plBau30lpgGVAY7vgbec/FWB9blwKL7e38SL5vIA/42r7nb4Df28dzgQXAGuBVIN4+nmDvr7Gfzw33PRzn/Z8BzIj0e7bvbYm9La/LVU7/butIW6WUihItvUlHKaVUkDThK6VUlNCEr5RSUUITvlJKRQlN+EopFSU04auoIiK19myFdVuTzbAqItnSYHZTpZqblj61glLHar8xJj/cQSgVDlrDV4r6ucofsuemXyAi3e3jXUXkY3tO8o9F5AT7eHsRedOex36JiJxmn8otIk/Zc9t/YI+eVapZ0ISvok3iEU06VzV4bo8xZgDwD6y5XbAfP2uMyQOmARPt4xOB2caax74Aa/QkWPOX/9MY0wfYBVwW4vtRKmg60lZFFRHZZ4xJ8XJ8A9ZCJOvsCdx+MMaki8gOrHnID9rHtxpjMkSkDMgyxlQ1OEc28KGxFrNARH4NxBpj/hj6O1MqMK3hK3WI8fHYVxlvqho8rkW/J1PNiCZ8pQ65qsHPufbjL7FmdAS4BphjP/4YuAXqFzBJcypIpRpLax8q2iTaq0vVed8YU9c1M15E5mNVhEbax+4AnhaRe4Ey4Gf28TuBSSIyDqsmfwvW7KZKNVvahq8U9W34hcaYHeGORalQ0SYdpZSKElrDV0qpKKE1fKWUihKa8JVSKkpowldKqSihCV8ppaKEJnyllIoSmvCVUipK/D9eUukVP8MBKwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7ffed5921c88>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7ffeec6b8978>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "\n",
    "# First, normalize training data\n",
    "meanI = train_featuresI.mean(axis=0)\n",
    "meanB = train_featuresB.mean(axis=0)\n",
    "stdI = train_featuresI.std(axis=0)\n",
    "stdB = train_featuresB.std(axis=0)\n",
    "trainI = (train_featuresI - meanI) / stdI\n",
    "trainB = (train_featuresB - meanB) / stdB\n",
    "testI = (test_featuresI - meanI) / stdI\n",
    "testB = (test_featuresB - meanB) / stdB\n",
    "\n",
    "\n",
    "def build_model(train_data):\n",
    "    model = keras.Sequential([\n",
    "        keras.layers.Dense(10, activation=tf.nn.relu,\n",
    "                          input_shape=(train_data.shape[1],)),\n",
    "        keras.layers.Dense(10, activation=tf.nn.relu),\n",
    "        keras.layers.Dense(1)\n",
    "    ])\n",
    "    \n",
    "    optimizer = tf.train.RMSPropOptimizer(0.001)\n",
    "    \n",
    "    model.compile(loss='mse',\n",
    "                 optimizer=optimizer,\n",
    "                 metrics=['mae'])\n",
    "    return model\n",
    "\n",
    "EPOCHS = 500\n",
    "modelI = build_model(train_featuresI)\n",
    "modelB = build_model(train_featuresB)\n",
    "modelI.summary()\n",
    "\n",
    "# Train model and store training stats\n",
    "historyI = modelI.fit(trainI, train_labelsI, epochs=EPOCHS,\n",
    "                      validation_split=0.2,verbose=0)\n",
    "historyB = modelB.fit(trainB, train_labelsB, epochs=EPOCHS,\n",
    "                      validation_split=0.2,verbose=0)\n",
    "\n",
    "def plot_history(history):\n",
    "    plt.figure()\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.ylabel('Mean Abs Error [DFMC]')\n",
    "    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),\n",
    "            label='Train Loss')\n",
    "    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),\n",
    "            label='Val Loss')\n",
    "    plt.legend()\n",
    "    #plt.ylim([0,5])\n",
    "    \n",
    "plot_history(historyI)\n",
    "plot_history(historyB)\n",
    "\n",
    "[lossI, maeI] = modelI.evaluate(testI,test_labelsI,verbose=0)\n",
    "print(\"Mean Absolute Error for Indices: \", maeI)\n",
    "[lossB, maeB] = modelB.evaluate(testB,test_labelsB,verbose=0)\n",
    "print(\"Mean Absolute Error for Reflectances: \", maeB)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define plot and save figure\n",
    "\n",
    "def plot_grid(grid_z,outfile):\n",
    "    grid_z_pos = grid_z[np.where(grid_z != -9999.0)]\n",
    "    print(\"min positive fuel moisture = %f, max positive fuel mositure = %f\" % (min(grid_z_pos),np.max(grid_z_pos)))\n",
    "    plt.matshow(grid_z,vmin=np.min(grid_z_pos)-1.0,vmax=np.max(grid_z_pos)+1.0,cmap='hot')\n",
    "    plt.colorbar()\n",
    "    if outfile:\n",
    "        plt.savefig(outfile) \n",
    "    else:\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'DataFrame' object has no attribute 'samples'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-5-cd24252a22fe>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#Height\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mstore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msamples\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m6\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;31m#Width\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mstore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msamples\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/pandas/core/generic.py\u001b[0m in \u001b[0;36m__getattr__\u001b[0;34m(self, name)\u001b[0m\n\u001b[1;32m   3612\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mname\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_info_axis\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3613\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3614\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mobject\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__getattribute__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3615\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3616\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__setattr__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'DataFrame' object has no attribute 'samples'"
     ]
    }
   ],
   "source": [
    "#Height\n",
    "store.samples[0][-6]\n",
    "#Width\n",
    "store.samples[0][-5]\n",
    "\n",
    "grid_z1 = store.root.samples[0][0]\n",
    "grid_z1 = grid_z1.reshape((683,872))\n",
    "plot_grid(grid_z1,None)\n",
    "\n",
    "grid_z2 = store.root.samples[1][0]\n",
    "grid_z2 = grid_z2.reshape((683,872))\n",
    "plot_grid(grid_z2,None)\n",
    "\n",
    "grid_z3 = store.root.samples[2][0]\n",
    "grid_z3 = grid_z3.reshape((683,872))\n",
    "plot_grid(grid_z3,None)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
