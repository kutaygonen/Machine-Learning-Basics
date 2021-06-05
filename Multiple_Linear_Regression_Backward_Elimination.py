# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:48:56 2020

@author: Kutay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

veriler = pd.read_csv('veriler.csv')
Yas = veriler.iloc[:,1:4].values
ulke = veriler.iloc[:,0:1].values
c = veriler.iloc[:,-1:].values

le= LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])

ohe=OneHotEncoder(categories='auto')
c=ohe.fit_transform(c).toarray()

ulke[:,0] = le.fit_transform(ulke[:,0])
ulke=ohe.fit_transform(ulke).toarray()


sonuc = pd.DataFrame(data = ulke, index = range(22) , columns = ['fr','tr' , 'us'])


sonuc2 = pd.DataFrame(data = Yas, index=range(22), columns = ['Boy', 'kilo', 'yaş'])


sonuc3 = pd.DataFrame(data = c[:,:1], index=range(22), columns = ['cinsiyet'])



# 2.4.1 - Data Frame Birleştirme - Concat ile
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)


s2=pd.concat([s,sonuc3] , axis=1)
print(s2)

# 2.5 - Veri bölmesi-Test ve Train(Eğitim)
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state=0)

# 2.6 - Verilerin Ölçeklendirmesi - Standardization

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)


x_train, x_test, y_train, y_test = train_test_split(veri,boy,test_size = 0.33, random_state=0)
r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred = r2.predict(x_test)

##BACKWARD ELIMINATION
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values= veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values

r=sm.OLS(endog = boy, exog = X_l).fit() #boy ile bağımsız değişkenler üzerindeki bağlantıyı kur
print(r.summary())

'''
import statsmodels.api as sm

model = sm.OLS(boy,X_l).fit()

print(model.summary())

'''

X = np.append(arr = np.ones((22,1)).astype(int), values= veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3,5]].values

r=sm.OLS(endog = boy, exog = X_l).fit() #boy ile bağımsız değişkenler üzerindeki bağlantıyı kur
print(r.summary())


X = np.append(arr = np.ones((22,1)).astype(int), values= veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3]].values

r=sm.OLS(endog = boy, exog = X_l).fit() #boy ile bağımsız değişkenler üzerindeki bağlantıyı kur
print(r.summary())






