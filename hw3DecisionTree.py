import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# utility to help you split training data
from sklearn.model_selection import train_test_split
# utility to standardize data http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler
# some dataset generation utilities. for example: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
from sklearn.datasets import make_moons, make_circles, make_classification

# Scoring for classifiers
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import pandas as pd

# Classifiers from scikit-learn
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Bayesian
from sklearn.naive_bayes import GaussianNB
# kNN
from sklearn.neighbors import KNeighborsClassifier
# DT
from sklearn.tree import DecisionTreeClassifier
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
# AdaBoost classifier (we talked about it today)
from sklearn.ensemble import AdaBoostClassifier
# KFold splitting
from sklearn.model_selection import KFold
# cross validation
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
# Jinesh Shah
# CSI 431
# HW # 3 Part B

# The names of the classifications
names = ["k=2",
         "k=5",
         "k=10",
         "k=20"
         ]
# Creating the classification for the Gini Decision Tree
classifiersDecisionTreeGini = [
     DecisionTreeClassifier(max_depth=2),
     DecisionTreeClassifier(max_depth=5),
     DecisionTreeClassifier(max_depth=10),
     DecisionTreeClassifier(max_depth=20)
    ]
# Creating the classification for the INF Decision Tree
classifiersDecisionTreeINF = [
     DecisionTreeClassifier(criterion="entropy",max_depth=2),
     DecisionTreeClassifier(criterion="entropy",max_depth=5),
     DecisionTreeClassifier(criterion="entropy",max_depth=10),
     DecisionTreeClassifier(criterion="entropy",max_depth=20)
]

# train dataset
data_train = pd.read_csv("cancer-data-train01.csv", header= None, usecols=[i for i in range(31)])
l = [i for i in range(30)]
X_train = data_train[l]
y_train = data_train[30]

# Cross Validation
# Creating the K Fold Cross Validation for it to be 10
cv = KFold(n_splits=10, shuffle= False, random_state=0)
# Decision Tree Gini
FmeasureGini = []
# working for all classifiers in the classifier array
# using cross validation to get the F measures from the classifiers
for clf in classifiersDecisionTreeGini:
    y_pred = cross_val_predict(clf, X_train,y_train, cv=cv)
    ap = average_precision_score(y_train,y_pred)
    rec = recall_score(y_train,y_pred, average="weighted")
    # calculating the F measure after getting the average precision and recall
    F1 = 2 * (ap*rec)/ (ap + rec)
    FmeasureGini += [F1]
# Show the results of the Gini Fmeasure
print(FmeasureGini)
# Decision Tree INF
FmeasureINF= []
# working for all classifiers in the classifier array
# using cross validation to get the F measures from the classifiers
for clf in classifiersDecisionTreeINF:
    y_pred = cross_val_predict(clf, X_train,y_train, cv=cv)
    ap = average_precision_score(y_train,y_pred)
    rec = recall_score(y_train,y_pred, average="weighted")
    # calculating the F measure after getting the average precision and recall
    F1 = 2 * (ap*rec)/ (ap + rec)
    FmeasureINF += [F1]
# Show the results of the INF Fmeasure
print(FmeasureINF)
# Plotting the results on a graph
# X axis is the names of the different k values for Gini
# Y axis is the F Measure
plt.plot(names,FmeasureGini)
plt.xlabel('K values for Decision Tree')
plt.ylabel('F Measure')
plt.title('Gini Graph')
plt.show()
# Plotting the results on a graph
# X axis is the names of the different k values for INF
# Y axis is the F Measure
plt.plot(names,FmeasureINF)
plt.xlabel('K values for Decision Tree')
plt.ylabel('F Measure')
plt.title('INF Graph')
plt.show()