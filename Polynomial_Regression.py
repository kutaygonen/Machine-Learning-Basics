# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:19:14 2020

@author: Kutay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import statsmodels.api as sm

# Veri Yükleme
veriler = pd.read_csv('maaslar.csv')

#data frema slicing
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPy array slicing
X= x.values
Y= y.values

#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

#Polynomial Regression
#Derece 4
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#nonlinear görselleştirme
plt.scatter(X,Y,color = 'green')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'red')
plt.grid()
plt.show()


#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))



















