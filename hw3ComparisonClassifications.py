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
# HW # 3 Part C and Extra Credit

# The names of the Classifier
names = ["C=0.01",
         "Gini k=5",
         "INF K=5"
         "LDA"
         "RFC"
         ]
# Creating the classifiers for all SVM, Decision Tree, LDA, and Random Forest Classifier
classifiers = [
    SVC(kernel="linear", C=0.01),
    DecisionTreeClassifier(max_depth=5),
    DecisionTreeClassifier(criterion="entropy",max_depth=5),
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001),
    RandomForestClassifier(n_estimators=100, max_depth= 2, random_state=0)
    ]

# train dataset
data_train = pd.read_csv("cancer-data-train01.csv", header= None, usecols=[i for i in range(31)])
l = [i for i in range(30)]
X_train = data_train[l]
y_train = data_train[30]

# test dataset
data_test = pd.read_csv("cancer-data-test01.csv", header= None, usecols=[i for i in range(31)])
l = [i for i in range(30)]
X_test = data_test[l]
y_test = data_test[30]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    # Train the classifier
    clf.fit(X_train, y_train)
    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    # Show value of average precision
    print(ap)
    rec = recall_score(y_test, y_pred, average='weighted')
    # Show value of average recall
    print(rec)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Show the value of the Fmeasures
    print(f1)

# Plotting the results on a graph
# X axis is the names of classifications
# Y axis is the Average Precision
plt.bar(names, ap)
plt.xlabel('Names of the Classifier')
plt.ylabel('Average Class Precision')
plt.title('Classifier Comparison Graph AP')
plt.show()

# Plotting the results on a graph
# X axis is the names of classifications
# Y axis is the Average Recall
plt.bar(names, rec)
plt.xlabel('Names of the Classifier')
plt.ylabel('Average Class Recall')
plt.title('Classifier Comparison Graph REC')
plt.show()
# Plotting the results on a graph
# X axis is the names of classifications
# Y axis is the F Measure
plt.bar(names, f1)
plt.xlabel('Names of the Classifier')
plt.ylabel('F Measure')
plt.title('Classifier Comparison Graph Fmeasure')
plt.show()

# NOTE: FOR SOME REASON I DON'T KNOW WHY BUT IT DOESN'T SHOW THE TWO EXTRA BARS FOR LDA AND
# RandomForestClassification. Probably messed up somewhere
