# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:56:13 2020

@author: Kutay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import statsmodels.api as sm
from sklearn.metrics import r2_score


# ----Veri Yükleme----
veriler = pd.read_csv('maaslar_yeni.csv')


#data frame slicing
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]

#NumPy array slicing
X= x.values
Y= y.values

#Correlation
print(veriler.corr())


#---Regressions----

#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

model = sm.OLS( lin_reg.predict(X), X ).fit()

print(model.summary())

print('Linear R2 degeri :')
print(r2_score(Y, lin_reg.predict(X)))


#Polynomial Regression
#Derece 4
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print('Poly OLS')
model2= sm.OLS( lin_reg2.predict(poly_reg.fit_transform(X)), X ).fit()
print(model2.summary())

print('Polynomial R2 degeri :')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


#SVR

#Scaling
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

x_scaled = sc1.fit_transform(X)
y_scaled = sc2.fit_transform(Y)


from sklearn.svm import SVR 

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_scaled ,y_scaled)

print('SVR OLS')
model3= sm.OLS( svr_reg.predict(x_scaled), x_scaled ).fit()
print(model3.summary())

print('SVR R2 degeri :')
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))


#Decesion Tree
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print('Decesion Tree OLS')
model4= sm.OLS(r_dt.predict(X), X).fit()
print(model4.summary())


print('Decesion Tree R2 degeri :')
print(r2_score(Y, r_dt.predict(X)))

#Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators =10 ,random_state=0)
rf_reg.fit(X,Y.ravel())

print('Random Forest OLS')
model5= sm.OLS(rf_reg.predict(X), X).fit()
print(model5.summary())

print('Random Forest R2 degeri :')
print(r2_score(Y, rf_reg.predict(X)))


#R2 yöntemi
# from sklearn.metrics import r2_score
print('Linear R2 degeri :')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2 degeri :')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 degeri :')
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))

print('Decesion Tree R2 degeri :')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 degeri :')
print(r2_score(Y, rf_reg.predict(X)))







