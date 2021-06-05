# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:41:09 2020

@author: Kutay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('Social_Network_Ads.csv')

X = veriler.iloc[:,[2,3]].values
Y= veriler.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train) #fit --> öğren transform --> uygula
X_test = sc.transform(x_test)

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', random_state=0)
svc.fit(X_train ,y_train)

y_pred_svc = svc.predict(X_test)

cm_svc = confusion_matrix(y_test, y_pred_svc)


#k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
'''
1. parametre = estimator : classifier
2. "      = X
3. "      = Y
4. "      = cv : kaç katmanlı
'''

cvs = cross_val_score(estimator=svc,X = X_train ,y = y_train, cv=4)
print(cvs.mean())
print(cvs.std()) #Standart Sapma ne kadar düşükse o kadar iyi

#Parametre Optimizasyonu ve Algoritma Seçimi
from sklearn.model_selection import GridSearchCV

p = [{'C': [1,2,3,4,5], 'kernel' : ['linear']},
     {'C' : [1,10,100,1000], 'kernel' : ['rbf'],
      'gamma' : [1,0.5,0.1,0.01,0.001]}] #GSCV nin aramak istediği parametreler


gs = GridSearchCV(estimator = svc,param_grid = p, scoring='accuracy', cv = 10 ,n_jobs =-1)

grid_search = gs.fit(X_train,y_train)
en_iyi_sonuc = grid_search.best_score_
en_iyi_parametreler = grid_search.best_params_

























