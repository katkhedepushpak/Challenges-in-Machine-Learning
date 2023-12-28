### AUTHOR : PUSHPAK VIJAY KATKHEDE
### AI 539 - ML Challenges
### Assignment 2: Give Your Models a Grade
### DATE : 02/04/2023


#importing required modules
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

#reading in the required files via pandas
data = pd.read_csv('activity-dev.csv')
test = pd.read_csv("activity-heldout.csv")

#defining the feature labels and target label on the datasets

X_train_full = data[['G_front', 'G_vert', 'G_lat', 'ant_id', 'RSSI', 'phase', 'freq']]
y_train_full = data['activity']


X_test_full = test[['G_front', 'G_vert', 'G_lat', 'ant_id', 'RSSI', 'phase', 'freq']]
y_test_full = test['activity']

#creating the classifier object

clf_mlp_f = MLPClassifier(random_state=9, max_iter=300)

clf_mlp_f.fit(X_train_full, y_train_full)

predicted = clf_mlp_f.predict(X_test_full)

print(len(predicted))

print(len(X_test_full))


#Geneating the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test_full, predicted, labels = clf_mlp_f.classes_)

#creating display object
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = clf_mlp_f.classes_)

#Displaying the confusion matrix
cm_display.plot()




