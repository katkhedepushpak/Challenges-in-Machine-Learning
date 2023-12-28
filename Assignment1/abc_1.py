### AUTHOR : PUSHPAK VIJAY KATKHEDE
### ASSIGNMENT 1: SWEESE CHEESE IN SPACE
### DATE : 01/21/2023

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import itertools
from sklearn.model_selection import train_test_split


# df = pd.read_csv('cfhtlens.csv ')

#reading the data inside a dataframe and then converting it in a numpy array
data = pd.read_csv('cfhtlens.csv ', na_values=[99, -99], usecols=["CLASS_STAR", "PSF_e1", "PSF_e2", "scalelength", "model_flux", "MAG_u", "MAG_g", "MAG_r", "MAG_i", "MAG_z"]) # Fill in na_values, usecols
Xy = data.to_numpy()

# Transforming the target label from continuous values to categorical value i.e. True and False as per the given situtation
X = Xy[:, 1:]
y = Xy[:, 0] >= 0.5 # What value? See above.
missing = np.sum(np.isnan(X), axis=1) > 0

X_use = X[~missing]
y_use = y[~missing]
X_train, X_test, y_train, y_test = train_test_split(X_use, y_use, train_size=3000, random_state=0)
X_test_full = np.concatenate((X_test, X[missing]))
y_test_full = np.concatenate((y_test, y[missing]))

#Generating Dataframe for all available splits of numpy arrays
dfX_use = pd.DataFrame(X_use)
dfy_use = pd.DataFrame(y_use)
dfX_train = pd.DataFrame(X_train)
dfy_train = pd.DataFrame(y_train)

dfX_test = pd.DataFrame(X_test)
dfy_test = pd.DataFrame(y_test)

dfX_test_full = pd.DataFrame(X_test_full)
dfy_test_full = pd.DataFrame(y_test_full)

dfX_miss = pd.DataFrame(X[missing])
dfy_miss = pd.DataFrame(y[missing])

# training a SVC classifier and fitting it to the given classifier data
clf = SVC()
clf.fit(X_train, y_train)


#D
mean_col = np.nanmean(X_train, axis = 1) #calculating a mean array for the features

indexes = np.where(np.isnan(X_test_full)) #indexes that had missing values
X_test_full_mean = X_test_full #auxillary array for preseving original array
X_test_full_mean[indexes] = np.take(mean_col, indexes[1]) #actual imputing

acc_for_D = clf.score(X_test_full_mean, y_test_full) # predicitng accuracy
print(acc_for_D)

ind = np.where(np.isnan(X[missing]))
XData_x_missing = X[missing]
XData_x_missing [ind] = np.take(mean_col, indexes[1])