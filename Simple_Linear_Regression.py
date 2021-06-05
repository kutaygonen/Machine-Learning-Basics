# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:26:30 2020

@author: Kutay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



veriler = pd.read_csv('satislar.csv')

# print(veriler)

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]
print(aylar)
print(satislar)



x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state=0)

# Verileri Scale Etme
# sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)


#Linear Reg. Model İnşaası
lr = LinearRegression()
lr.fit(x_train,y_train)


tahmin=lr.predict(x_test) #Y_test i tahimn ediyor



x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)

plt.plot(x_test, tahmin)
plt.title('aylara göre satis')
plt.xlabel('aylar')
plt.ylabel('satislar')



