### AUTHOR : PUSHPAK VIJAY KATKHEDE
### AI 539 - ML Challenges
### Assignment 3: Adapt to Change
### DATE : 02/23/2023

#importing required modules
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from label_shift_adaptation import analyze_val_data, update_probs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#reading in the required files via pandas
train = pd.read_csv('train-TX.csv')
val = pd.read_csv("val-TX.csv")
test1 = pd.read_csv("test1-TX.csv")
test2 = pd.read_csv("test2-FL.csv")
test3 = pd.read_csv("test3-FL.csv")

#defining the feature labels and target label

X_t= train[["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]]
y_t= train[['Severity']]

X_v= val[["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]]
y_v= val[['Severity']]

X_t1= test1[["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]]
y_t1= test1[['Severity']]

X_t2= test2[["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]]
y_t2= test2[['Severity']]

X_t3= test3[["Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]]
y_t3= test3[['Severity']]

#Calculate Unique Class labels

classes = np.unique(y_t)

#defining classifiers

clf_dumMF = DummyClassifier(strategy='most_frequent', random_state=0)
clf_dumCD = DummyClassifier(strategy='stratified', random_state=0)
clf_ranFor = RandomForestClassifier(max_depth=3, random_state=0)
clf_gp = GaussianProcessClassifier(random_state=0)
clf_knnc3 =KNeighborsClassifier(n_neighbors=3)
clf_knnc9 =KNeighborsClassifier(n_neighbors=9)

#training the classifiers

clf_dumMF.fit(X_t,y_t)
clf_dumCD.fit(X_t,y_t)
clf_ranFor.fit(X_t,y_t)
clf_gp.fit(X_t,y_t)
clf_knnc3.fit(X_t,y_t)
clf_knnc9.fit(X_t,y_t)

#Validation Scores

bef_val_dumMF = clf_dumMF.predict(X_v)
bef_val_dumCD = clf_dumCD.predict(X_v)
bef_val_ranFor = clf_ranFor.predict(X_v)
bef_val_gp = clf_gp.predict(X_v)
bef_val_knnc3 = clf_knnc3.predict(X_v)
bef_val_knnc9 = clf_knnc9.predict(X_v)

acc_dumMF_val=accuracy_score(y_v, bef_val_dumMF, normalize=True)
acc_dumCD_val = accuracy_score(y_v, bef_val_dumCD, normalize=True)
acc_ranFor_val = accuracy_score(y_v, bef_val_ranFor, normalize=True)
acc_gp_val = accuracy_score(y_v, bef_val_gp, normalize=True)
acc_knnc3_val = accuracy_score(y_v, bef_val_knnc3, normalize=True)
acc_knnc9_val = accuracy_score(y_v, bef_val_knnc9, normalize=True)


acc_dumMF_val = str(round(acc_dumMF_val, 2))
acc_dumCD_val = str(round(acc_dumCD_val, 2))
acc_ranFor_val = str(round(acc_ranFor_val, 2))
acc_gp_val = str(round(acc_gp_val, 2))
acc_knnc3_val = str(round(acc_knnc3_val, 2))
acc_knnc9_val = str(round(acc_knnc9_val, 2))


#Test1 Scores

# Predicting the values for Test1 set
bef_t1_dumMF = clf_dumMF.predict(X_t1)
bef_t1_dumCD = clf_dumCD.predict(X_t1)
bef_t1_ranFor = clf_ranFor.predict(X_t1)
bef_t1_gp = clf_gp.predict(X_t1)
bef_t1_knnc3 = clf_knnc3.predict(X_t1)
bef_t1_knnc9 = clf_knnc9.predict(X_t1)

# Calculating the accuracy for Test1 set before BBSC
acc_dumMF_t1 = accuracy_score(y_t1, bef_t1_dumMF, normalize=True)
acc_dumCD_t1 = accuracy_score(y_t1, bef_t1_dumCD, normalize=True)
acc_ranFor_t1 = accuracy_score(y_t1, bef_t1_ranFor, normalize=True)
acc_gp_t1 = accuracy_score(y_t1, bef_t1_gp, normalize=True)
acc_knnc3_t1 = accuracy_score(y_t1, bef_t1_knnc3, normalize=True)
acc_knnc9_t1 = accuracy_score(y_t1, bef_t1_knnc9, normalize=True)

# Calculating the old prediction probability for Test1 set after BBSC
prev_prob_t1_ranFor = clf_ranFor.predict_proba(X_t1)
prev_prob_t1_gp = clf_gp.predict_proba(X_t1)
prev_prob_t1_knnc3 = clf_knnc3.predict_proba(X_t1)
prev_prob_t1_knnc9 = clf_knnc9.predict_proba(X_t1)

# Calculating the weights for Test1 set by BBSC
weights_ranFor_t1 = analyze_val_data(y_v, bef_val_ranFor, bef_t1_ranFor)
weights_gp_t1 = analyze_val_data(y_v, bef_val_gp, bef_t1_gp)
weights_knnc3_t1 = analyze_val_data(y_v, bef_val_knnc3, bef_t1_knnc3)
weights_knnc9_t1 = analyze_val_data(y_v, bef_val_knnc9, bef_t1_knnc9)

# Calculating the new prediction probability for Test1 set after BBSC
new_prob_t1_ranFor = update_probs(classes, weights_ranFor_t1, bef_t1_ranFor, prev_prob_t1_ranFor)
new_prob_t1_gp = update_probs(classes, weights_gp_t1, bef_t1_gp, prev_prob_t1_gp)
new_prob_t1_knnc3 = update_probs(classes, weights_knnc3_t1, bef_t1_knnc3, prev_prob_t1_knnc3)
new_prob_t1_knnc9 = update_probs(classes, weights_knnc9_t1, bef_t1_knnc9, prev_prob_t1_knnc9)

# Calculating the accuracy for Test1 set after BBSC
acc_ranFor_t1_after = accuracy_score(y_t1, new_prob_t1_ranFor[0], normalize=True)
acc_gp_t1_after = accuracy_score(y_t1, new_prob_t1_gp[0], normalize=True)
acc_knnc3_t1_after = accuracy_score(y_t1, new_prob_t1_knnc3[0], normalize=True)
acc_knnc9_t1_after = accuracy_score(y_t1, new_prob_t1_knnc9[0], normalize=True)

# Rounding off the values to the 2nd decimal
acc_dumMF_t1 = str(round(acc_dumMF_t1, 2))
acc_dumCD_t1 = str(round(acc_dumCD_t1, 2))
acc_ranFor_t1 = str(round(acc_ranFor_t1, 2))
acc_gp_t1 = str(round(acc_gp_t1, 2))
acc_knnc3_t1 = str(round(acc_knnc3_t1, 2))
acc_knnc9_t1 = str(round(acc_knnc9_t1, 2))

weights_ranFor_t1 = ['%.2f' % elem for elem in weights_ranFor_t1]
weights_gp_t1 = ['%.2f' % elem for elem in weights_gp_t1]
weights_knnc3_t1 = ['%.2f' % elem for elem in weights_knnc3_t1]
weights_knnc9_t1 = ['%.2f' % elem for elem in weights_knnc9_t1]

acc_ranFor_t1_after = str(round(acc_ranFor_t1_after, 2))
acc_gp_t1_after = str(round(acc_gp_t1_after, 2))
acc_knnc3_t1_after = str(round(acc_knnc3_t1_after, 2))
acc_knnc9_t1_after = str(round(acc_knnc9_t1_after, 2))

# Test2 Scores

# Predicting the values for Test2 set
bef_t2_dumMF = clf_dumMF.predict(X_t2)
bef_t2_dumCD = clf_dumCD.predict(X_t2)
bef_t2_ranFor = clf_ranFor.predict(X_t2)
bef_t2_gp = clf_gp.predict(X_t2)
bef_t2_knnc3 = clf_knnc3.predict(X_t2)
bef_t2_knnc9 = clf_knnc9.predict(X_t2)

# Calculating the accuracy for Test2 set before BBSC
acc_dumMF_t2 = accuracy_score(y_t2, bef_t2_dumMF, normalize=True)
acc_dumCD_t2 = accuracy_score(y_t2, bef_t2_dumCD, normalize=True)
acc_ranFor_t2 = accuracy_score(y_t2, bef_t2_ranFor, normalize=True)
acc_gp_t2 = accuracy_score(y_t2, bef_t2_gp, normalize=True)
acc_knnc3_t2 = accuracy_score(y_t2, bef_t2_knnc3, normalize=True)
acc_knnc9_t2 = accuracy_score(y_t2, bef_t2_knnc9, normalize=True)

# Calculating the prediction probability for Test2 set before BBSC
prev_prob_t2_ranFor = clf_ranFor.predict_proba(X_t2)
prev_prob_t2_gp = clf_gp.predict_proba(X_t2)
prev_prob_t2_knnc3 = clf_knnc3.predict_proba(X_t2)
prev_prob_t2_knnc9 = clf_knnc9.predict_proba(X_t2)

# Calculating the weights for Test2 set by BBSC
weights_ranFor_t2 = analyze_val_data(y_v, bef_val_ranFor, bef_t2_ranFor)
weights_gp_t2 = analyze_val_data(y_v, bef_val_gp, bef_t2_gp)
weights_knnc3_t2 = analyze_val_data(y_v, bef_val_knnc3, bef_t2_knnc3)
weights_knnc9_t2 = analyze_val_data(y_v, bef_val_knnc9, bef_t2_knnc9)

# Calculating the new prediction probability for Test2 set after BBSC
new_prob_t2_ranFor = update_probs(classes, weights_ranFor_t2, bef_t2_ranFor, prev_prob_t2_ranFor)
new_prob_t2_gp = update_probs(classes, weights_gp_t2, bef_t2_gp, prev_prob_t2_gp)
new_prob_t2_knnc3 = update_probs(classes, weights_knnc3_t2, bef_t2_knnc3, prev_prob_t2_knnc3)
new_prob_t2_knnc9 = update_probs(classes, weights_knnc9_t2, bef_t2_knnc9, prev_prob_t2_knnc9)

# Calculating the accuracy for Test2 set after BBSC
acc_ranFor_t2_after = accuracy_score(y_t2, new_prob_t2_ranFor[0], normalize=True)
acc_gp_t2_after = accuracy_score(y_t2, new_prob_t2_gp[0], normalize=True)
acc_knnc3_t2_after = accuracy_score(y_t2, new_prob_t2_knnc3[0], normalize=True)
acc_knnc9_t2_after = accuracy_score(y_t2, new_prob_t2_knnc9[0], normalize=True)

# Rounding off the values to the 2nd decimal
acc_dumMF_t2 = str(round(acc_dumMF_t2, 2))
acc_dumCD_t2 = str(round(acc_dumCD_t2, 2))
acc_ranFor_t2 = str(round(acc_ranFor_t2, 2))
acc_gp_t2 = str(round(acc_gp_t2, 2))
acc_knnc3_t2 = str(round(acc_knnc3_t2, 2))
acc_knnc9_t2 = str(round(acc_knnc9_t2, 2))

weights_ranFor_t2 = ['%.2f' % elem for elem in weights_ranFor_t2]
weights_gp_t2 = ['%.2f' % elem for elem in weights_gp_t2]
weights_knnc3_t2 = ['%.2f' % elem for elem in weights_knnc3_t2]
weights_knnc9_t2 = ['%.2f' % elem for elem in weights_knnc9_t2]

acc_ranFor_t2_after = str(round(acc_ranFor_t2_after, 2))
acc_gp_t2_after = str(round(acc_gp_t2_after, 2))
acc_knnc3_t2_after = str(round(acc_knnc3_t2_after, 2))
acc_knnc9_t2_after = str(round(acc_knnc9_t2_after, 2))

# Test3 Scores

# Predicting the values for Test3 set
bef_t3_dumMF = clf_dumMF.predict(X_t3)
bef_t3_dumCD = clf_dumCD.predict(X_t3)
bef_t3_ranFor = clf_ranFor.predict(X_t3)
bef_t3_gp = clf_gp.predict(X_t3)
bef_t3_knnc3 = clf_knnc3.predict(X_t3)
bef_t3_knnc9 = clf_knnc9.predict(X_t3)

# Calculating the accuracy for Test3 set before BBSC
acc_dumMF_t3 = accuracy_score(y_t3, bef_t3_dumMF, normalize=True)
acc_dumCD_t3 = accuracy_score(y_t3, bef_t3_dumCD, normalize=True)
acc_ranFor_t3 = accuracy_score(y_t3, bef_t3_ranFor, normalize=True)
acc_gp_t3 = accuracy_score(y_t3, bef_t3_gp, normalize=True)
acc_knnc3_t3 = accuracy_score(y_t3, bef_t3_knnc3, normalize=True)
acc_knnc9_t3 = accuracy_score(y_t3, bef_t3_knnc9, normalize=True)

# Calculating the prediction probability for Test3 set before BBSC
clf_ranFor.predict_proba(X_t3)
clf_gp.predict_proba(X_t3)
clf_knnc3.predict_proba(X_t3)
clf_knnc9.predict_proba(X_t3)

# Calculating the weights for Test3 set by BBSC
weights_ranFor_t3 = analyze_val_data(y_v, bef_val_ranFor, bef_t3_ranFor)
weights_gp_t3 = analyze_val_data(y_v, bef_val_gp, bef_t3_gp)
weights_knnc3_t3 = analyze_val_data(y_v, bef_val_knnc3, bef_t3_knnc3)
weights_knnc9_t3 = analyze_val_data(y_v, bef_val_knnc9, bef_t3_knnc9)

# Calculating the new prediction probability for Test3 set after BBSC
prev_prob_t3_ranFor = clf_ranFor.predict_proba(X_t3)
prev_prob_t3_gp = clf_gp.predict_proba(X_t3)
prev_prob_t3_knnc3 = clf_knnc3.predict_proba(X_t3)
prev_prob_t3_knnc9 = clf_knnc9.predict_proba(X_t3)

# Calculating the accuracy for Test3 set after BBSC
new_prob_t3_ranFor = update_probs(classes, weights_ranFor_t3, bef_t2_ranFor, prev_prob_t3_ranFor)
new_prob_t3_gp = update_probs(classes, weights_gp_t3, bef_t3_gp, prev_prob_t3_gp)
new_prob_t3_knnc3 = update_probs(classes, weights_knnc3_t3, bef_t3_knnc3, prev_prob_t3_knnc3)
new_prob_t3_knnc9 = update_probs(classes, weights_knnc9_t3, bef_t3_knnc9, prev_prob_t3_knnc9)

# Rounding off the values to the 2nd decimal
acc_ranFor_t3_after = accuracy_score(y_t3, new_prob_t3_ranFor[0], normalize=True)
acc_gp_t3_after = accuracy_score(y_t3, new_prob_t3_gp[0], normalize=True)
acc_knnc3_t3_after = accuracy_score(y_t3, new_prob_t3_knnc3[0], normalize=True)
acc_knnc9_t3_after = accuracy_score(y_t3, new_prob_t3_knnc9[0], normalize=True)

# Rounding off the values to the 2nd decimal
acc_dumMF_t3 = str(round(acc_dumMF_t3, 2))
acc_dumCD_t3 = str(round(acc_dumCD_t3, 2))
acc_ranFor_t3 = str(round(acc_ranFor_t3, 2))
acc_gp_t3 = str(round(acc_gp_t3, 2))
acc_knnc3_t3 = str(round(acc_knnc3_t3, 2))
acc_knnc9_t3 = str(round(acc_knnc9_t3, 2))

weights_ranFor_t3 = ['%.2f' % elem for elem in weights_ranFor_t3]
weights_gp_t3 = ['%.2f' % elem for elem in weights_gp_t3]
weights_knnc3_t3 = ['%.2f' % elem for elem in weights_knnc3_t3]
weights_knnc9_t3 = ['%.2f' % elem for elem in weights_knnc9_t3]

acc_ranFor_t3_after = str(round(acc_ranFor_t3_after, 2))
acc_gp_t3_after = str(round(acc_gp_t3_after, 2))
acc_knnc3_t3_after = str(round(acc_knnc3_t3_after, 2))
acc_knnc9_t3_after = str(round(acc_knnc9_t3_after, 2))

# Tabulating
Table1 = [["Accuracy", "Val-TX", "Test1-TX", "Test2-FL", "Test3-FL"],
          ["Baseline: most_frequent",acc_dumMF_val,acc_dumMF_t1,acc_dumMF_t2,acc_dumMF_t3],
          ["Baseline: stratified",acc_dumCD_val,acc_dumCD_t1,acc_dumCD_t2,acc_dumCD_t3],
          ["RandomForest",acc_ranFor_val, [acc_ranFor_t1,acc_ranFor_t1_after], [acc_ranFor_t2,acc_ranFor_t2_after],[acc_ranFor_t3,acc_ranFor_t3_after]],
          ["GaussianProcess",acc_gp_val, [acc_gp_t1,acc_gp_t1_after], [acc_gp_t2,acc_gp_t2_after],[acc_gp_t3,acc_gp_t3_after]],
          ["3-NearestNeighbor",acc_knnc3_val,[acc_knnc3_t1,acc_knnc3_t1_after], [acc_knnc3_t2,acc_knnc3_t2_after],[acc_knnc3_t3,acc_knnc3_t3_after]],
          ["9-NearestNeighbor",acc_knnc9_val,[acc_knnc9_t1,acc_knnc9_t1_after], [acc_knnc9_t2,acc_knnc9_t2_after],[acc_knnc9_t3,acc_knnc9_t3_after]]]

print("1. Accuracy comparison table before and after the BBSC \n")
print(tabulate(Table1, headers="firstrow", tablefmt="presto"))


Table2 = [["Accuracy", "Test1-TX", "Test2-FL", "Test3-FL"],
          ["RandomForest",weights_ranFor_t1, weights_ranFor_t2, weights_ranFor_t3],
          ["GaussianProcess",weights_gp_t1, weights_gp_t2, weights_gp_t3],
          ["3-NearestNeighbor",weights_knnc3_t1, weights_knnc3_t2, weights_knnc3_t3],
          ["9-NearestNeighbor",weights_knnc9_t1, weights_knnc9_t2, weights_knnc9_t3]]

print("\n\n-------------------------------------------------------------------------------------------")
print("2. Adaptation weights calculated through  BBSC  \n\n")
print(tabulate(Table2, headers="firstrow", tablefmt="presto"))
print("\n\n-------------------------------------------------------------------------------------------")

# Plotting the grouped Bar plot

# Normalizing the representation of classes between 0-1
val_true_dist = val.groupby('Severity').size()
t1_true_dist = test1.groupby('Severity').size()
t2_true_dist = test2.groupby('Severity').size()
t3_true_dist = test3.groupby('Severity').size()
val_true_dist = val_true_dist.tolist()
t1_true_dist = t1_true_dist.tolist()
t2_true_dist = t2_true_dist.tolist()
t3_true_dist = t3_true_dist.tolist()

val_size = sum(val_true_dist)
t1_size = sum(t1_true_dist)
t2_size = sum(t2_true_dist)
t3_size = sum(t3_true_dist)

val_true_dist[:] = [x / val_size for x in val_true_dist]
t1_true_dist[:] = [x / t1_size for x in t1_true_dist]
t2_true_dist[:] = [x / t2_size for x in t2_true_dist]
t3_true_dist[:] = [x / t3_size for x in t3_true_dist]


# generating a bar plot
classes_1 = ("Class 1", "Class 2", "Class 3", 'Class 4')
distribution = {
    'val': (val_true_dist[0], val_true_dist[1], val_true_dist[2], val_true_dist[3]),
    'test1': (t1_true_dist[0], t1_true_dist[1], t1_true_dist[2], t1_true_dist[3]),
    'test2': (t2_true_dist[0], t2_true_dist[1], t2_true_dist[2], t2_true_dist[3]),
    'test3': (t3_true_dist[0], t3_true_dist[1], t3_true_dist[2], t3_true_dist[3]),
}

x = np.arange(len(classes_1))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(constrained_layout=True)
for attribute, measurement in distribution.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Class Count')
ax.set_title('Class Distribution in Test Sets')
ax.set_xticks(x + width, classes_1)
ax.legend(loc='upper right')
ax.set_ylim(0, 1)
plt.show()


# Calculating Feature Importance for random Forest Classifier
importances = clf_ranFor.feature_importances_
columns = X_t.columns
feature_imp_df = list(zip(columns, importances))
feature_imp_df = pd.DataFrame(feature_imp_df, columns=['Feature','Importance'])
sorted_imp_fea = feature_imp_df.sort_values(by=['Importance'],ascending=False)
print("	Ranked Table as per the feature Importance for classifier Random Forest used above.")
print(tabulate(sorted_imp_fea, tablefmt="presto"))

print("\n\n-------------------------------------------------------------------------------------------")

#Draw Confusion Matrices over validation set

# for random forest
cm1 = metrics.confusion_matrix(val[['Severity']] , bef_val_ranFor)
print("CF for Random Forst")
print(cm1)

'''
[[ 23   0   0   0]
 [  3   2  73   0]
 [  1   0 185   0]
 [  0   0  56   1]]

'''
print("\nCF for Gaussian process")
# for GP
cm2 = metrics.confusion_matrix(val[['Severity']] , bef_val_gp)
print(cm2)

'''[[ 19   2   1   1]
 [  2  29  39   8]
 [  1  38 127  20]
 [  1   8  29  19]]
<matplotlib.image.Axe
'''


# For KNNC3
cm3 = metrics.confusion_matrix(val[['Severity']] , bef_val_knnc3)
print("\nCF for 3-KKNC")
print(cm3)

'''
[[ 20   2   0   1]
 [  2  31  41   4]
 [  2  50 118  16]
 [  1  11  37   8]]
'''

# For KNNC9
cm4 = metrics.confusion_matrix(val[['Severity']] , bef_val_knnc9)
print("\nCF for 9-KKNC")
print(cm4)

'''
[[ 17   3   1   2]
 [  2  26  49   1]
 [  0  27 152   7]
 [  1   4  47   5]]
'''