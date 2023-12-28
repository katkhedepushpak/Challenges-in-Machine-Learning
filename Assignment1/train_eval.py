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

#A Abstaining the missing values
acc_a = clf.score(X_test, y_test)
print(acc_a) #this is the accuracy for  dataset that doesnt contains any missing value

no_correct_pred = acc_a * len(y_test) #calculating the total no of correct predicitions
new_accuracy_a = no_correct_pred / len(y_test_full) # Adjusting for abstained value as errors and hence taking correct preds accuracy out of fulll dataset
print(new_accuracy_a) #Final Acuracy


#B Counting the number of occurences of classes to identify a majority class
count = dfy_train.groupby(0).size().to_dict()
mclass_count = max(count[0], count[1])

majority_class = [k for k, v in count.items() if v == mclass_count]

#Adding Predicition for all classes with values equal to majority class
assign_majority = [majority_class for x in y[missing]]
merged = list(itertools.chain(*assign_majority))


missing_test_arr = y[missing]
missing_test = missing_test_arr.tolist()

count_pred_corr = 0
for i in missing_test:
    if merged[i] == missing_test[i]:
        count_pred_corr = count_pred_corr + 1
accuracy_with_missing = count_pred_corr / len(missing_test)
print(accuracy_with_missing)     # this is accuracy for prediciton on missing values set only
acc_clf_b = clf.score(X_test, y_test)
no_of_crr_by_clf = len(X_test) * acc_clf_b
new_acc_b = (count_pred_corr + no_of_crr_by_clf) / len(y_test_full)
print(new_acc_b) # this is accuracy for prediciton on full data set

#C Omitting the features with missing values
#Adjusting numoy arrays into dataframe for manpulation purpose
X_full = np.concatenate((X_train, X_test_full))
dfX_full = pd.DataFrame(X_full)
y_full = np.concatenate((y_train, y_test_full))
dfy_full = pd.DataFrame(y_full)

#dropped features with any missing values completely
new_x = dfX_full.dropna(axis = 1)
dfy_fulln = dfy_full[0]

#Split the new obtained dataset for training with only 4 features
CX_train, CX_test, Cy_train, Cy_test = train_test_split(new_x, dfy_fulln, train_size=3000, random_state=0)
model_c = SVC()
model_c.fit(CX_train, Cy_train)

# achieved accurracy over new model as no missing data will be present 
acc_for_c = model_c.score(CX_test, Cy_test)
print(acc_for_c)

#converting x[missing] into 4 features
new_test_missing_c = pd.DataFrame(X[missing])
missing_test_c_adjusted = new_test_missing_c.dropna(axis = 1)

#testing accuracy for test set whose values were originally missing
acc_for_c_missing = model_c.score(missing_test_c_adjusted, y[missing])
print(acc_for_c_missing)


#D  Imouting using a mean value
mean_col = np.nanmean(X_train, axis = 1) #calculating a mean array for the features

indexes = np.where(np.isnan(X_test_full)) #indexes that had missing values
X_test_full_mean = X_test_full #auxillary array for preseving original array
X_test_full_mean[indexes] = np.take(mean_col, indexes[1]) #actual imputing

acc_for_D = clf.score(X_test_full_mean, y_test_full) # predicitng accuracy
print(acc_for_D)

#testing for only missing values set 
ind = np.where(np.isnan(X[missing]))
XData_x_missing = X[missing]
XData_x_missing [ind] = np.take(mean_col, indexes[1])
acc_for_D_m = clf.score(XData_x_missing, y[missing]) #accuracy with onlymissing values set
print(acc_for_D_m)

#E Folowing Median as data is more concentrated around median
indexes = np.where(np.isnan(X_test_full)) 
median_col = np.median(X_train, axis=1) #calculating a median array for the features
X_test_full_median = X_test_full
X_test_full_median[indexes] = np.take(median_col, indexes[1])
acc_for_E= clf.score(X_test_full_median, y_test_full)
print(acc_for_E)

#testing for only missing values set
ind = np.where(np.isnan(X[missing]))
XData_x_missing_median = X[missing]
XData_x_missing_median [ind] = np.take(median_col, indexes[1])
acc_for_E_m = clf.score(XData_x_missing_median, y[missing]) #accuracy with onlymissing values set
print(acc_for_E_m)

#I have removed the objects last features as I am trying to predict using the classifier trained in Method C which omitted features with missing values
own_obj = [[0.0191, 0, 0.3947, 13.2]]
pred = model_c.predict(own_obj)
print(pred)

#output is False which is correct
new_own = [[0.0191, 0, 0.3947, 13.2, 23.919,23.773, 23.407, 22.87, 22.495]]
pred1 = clf.predict(new_own)
print(pred1)
#Using other classifier also we were able to gain the correct answer for our object.