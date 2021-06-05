# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:17:37 2020

@author: Kutay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 - Veri Önişleme

# 2.1 - Veri Yükleme
veriler = pd.read_csv('Churn_Modelling.csv')

Y = veriler.iloc[:,13].values #Bağımlı Değişken

# 2.3 - Encoder - Kategorik -> Numerik
ulke = veriler.iloc[:,4:5].values
cinsiyet = veriler.iloc[:,5:6].values
kredi = veriler.iloc[:,3:4].values
geri_kalan = veriler.iloc[:,6:13].values

# 2.3.1 Label Encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

ohe=OneHotEncoder(categories='auto')
ulke=ohe.fit_transform(ulke).toarray()

le2 = LabelEncoder()
cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])

ulke_new = ulke[:,0:2]

ulke_kolon = pd.DataFrame(data = ulke_new, index = range(10000) , columns = ['fr','gr'])
cinsiyet_kolon = pd.DataFrame(data = cinsiyet, index = range(10000) , columns = ['Gender'])
kredi_kolon = pd.DataFrame(data = kredi, index = range(10000) , columns = ['CreditScore'])
geri_kalan_kolon = pd.DataFrame(data = geri_kalan, index = range(10000) , columns = ['Age' , 'Tenure', 'Balance','NumOfProducts','HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

s=pd.concat([ulke_kolon,kredi_kolon],axis=1)
s1=pd.concat([s,cinsiyet_kolon],axis=1)
X = pd.concat([s1,geri_kalan_kolon],axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Yapay Sinir Aglari

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu', input_dim = 11))

classifier.add(Dense(6,kernel_initializer ='uniform',activation = 'relu'))

classifier.add(Dense(1,kernel_initializer ='uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, epochs=50)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train,y_train)

y_pred_xgb = xgb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred_xgb,y_test)



















