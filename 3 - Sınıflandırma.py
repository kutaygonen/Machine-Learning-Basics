# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:37:51 2020

@author: Kutay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 - Veri Önişleme

# 2.1 - Veri Yükleme
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4] #bağımsız değişken
y = veriler.iloc[:,4:] #bağımlı değişken

X= x.values
Y= y.values


# 2.5 - Veri bölmesi-Test ve Train(Eğitim)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state=0)

# 2.6 - Verilerin Ölçeklendirmesi - Standardization

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()

X_train = sc.fit_transform(x_train) #fit --> öğren transform --> uygula
X_test = sc.transform(x_test)

#-----SINIFLANDIRMA ALGORITMALARI----

# 1- Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)

logr.fit(X_train , y_train) #eğitim

y_pred = logr.predict(X_test)


#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)



# 2- KNN -K Nearest

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1 , metric= 'minkowski')
knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)

# 3- (SVC) SVM - Support Vector Machine

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')
svc.fit(X_train ,y_train)

y_pred_svc = svc.predict(X_test)

cm_svc = confusion_matrix(y_test, y_pred_svc)


# 4- Naive Bayes

from sklearn.naive_bayes import BernoulliNB

gnb = BernoulliNB()

gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)

cm_gnb = confusion_matrix(y_test, y_pred_gnb)


# 5- Decesion Tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion= 'entropy')
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

cm_dtc = confusion_matrix(y_test, y_pred_dtc)

# 6- Random Forest Sınıflandırması

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10 , criterion= 'entropy')

rfc.fit(X_train , y_train)

y_pred_rfc = rfc.predict(X_test)

cm_rfc = confusion_matrix(y_test, y_pred_rfc)

#Yüzde oranları AUC ROC
y_proba = rfc.predict_proba(X_test)

# -- ROC HESABI --
#TPR,FPR 
from sklearn import metrics

fpr,tpr,thold = metrics.roc_curve(y_test, y_proba[:,0] ,pos_label = 'e' )


