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
# HW # 3 Part A

# The names of the classifiers for svm
names = ["0.01",
         "0.1",
         "1",
         "10",
         "100"
         ]
# Creating the classifiers
classifiers = [
    SVC(kernel="linear", C=0.01),
    SVC(kernel="linear", C=0.1),
    SVC(kernel="linear", C=1),
    SVC(kernel="linear", C=10),
    SVC(kernel="linear", C=100),
    ]

# train dataset
data_train = pd.read_csv("cancer-data-train01.csv", header= None, usecols=[i for i in range(31)])
l = [i for i in range(30)]
X_train = data_train[l]
y_train = data_train[30]


# Cross validation
# KFold for 10 splits
cv = KFold(n_splits=10, shuffle= False, random_state=0)
Fmeasure = []
# working for all classifiers in the classifier array
# using cross validation to get the F measures from the classifiers
for clf in classifiers:
    y_pred = cross_val_predict(clf, X_train,y_train, cv=cv)
    ap = average_precision_score(y_train,y_pred)
    rec = recall_score(y_train,y_pred, average="weighted")
    # calculating the F measure after getting the average precision and recall
    F1 = 2 * (ap*rec)/ (ap + rec)
    Fmeasure += [F1]
# Print out the values of the F measure here
print(Fmeasure)

# Plotting the results on a graph
# X axis is the names of the different C values for SVM
# Y axis is the F Measure
plt.plot(names, Fmeasure)
plt.xlabel('C values for SVM')
plt.ylabel('F Measure')
plt.title('SVM Graph')
plt.show()