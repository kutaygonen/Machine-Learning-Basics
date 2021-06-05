# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:57:42 2020

@author: Kutay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 - Veri Önişleme

# 2.1 - Veri Yükleme
veriler = pd.read_csv('Wine.csv')

X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:,13].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)


from sklearn.preprocessing import StandardScaler


sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#PCA - Principal Component Analaysis

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_2 = pca.fit_transform(X_train)
X_test_2 = pca.transform(X_test)
 
#Herhangi bir ML algoritması kullanılabilir
#Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0) #her seferinde aynı değer ise random_state = 0
classifier.fit(X_train,y_train)
y_pred_1 = classifier.predict(X_test)

classifier_2 = LogisticRegression(random_state=0)
classifier_2.fit(X_train_2,y_train)
y_pred_pca = classifier_2.predict(X_test_2)

#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_1 = confusion_matrix(y_test,y_pred_1)
cm_pca = confusion_matrix(y_test,y_pred_pca)
cm_both = confusion_matrix(y_pred_1,y_pred_pca)

#LDA - Linear Discreminant Analaysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2, )
X_train_lda = lda.fit_transform(X_train,y_train) #sınıflar arası fark aldığı için iki parametre
X_test_lda = lda.transform(X_test)

classifier_3 = LogisticRegression(random_state=0)
classifier_3.fit(X_train_lda,y_train)
y_pred_lda = classifier_3.predict(X_test_lda)
cm_lda = confusion_matrix(y_test,y_pred_lda)















