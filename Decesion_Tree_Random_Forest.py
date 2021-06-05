# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:38:48 2020

@author: Kutay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
plt.scatter(X,Y,color = 'yellow')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'red')
plt.grid()
plt.show()


#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

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
plt.scatter(x_scaled,y_scaled)
plt.plot(x_scaled, svr_reg.predict(x_scaled))
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


#Decsion Tree
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z=X+0.5
K=X-0.5
plt.scatter(X,Y, color = 'black')
plt.plot(x, r_dt.predict(X), color = 'magenta')
plt.plot(x, r_dt.predict(Z),color = 'red')
plt.plot(x, r_dt.predict(K), color = 'blue')
plt.show()
print('Decesion Tree')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

#Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators =10 ,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))
plt.scatter(X,Y, color = 'red')
plt.plot(x, rf_reg.predict(X), color = 'magenta')
# plt.plot(x, rf_reg.predict(Z), color = 'green')
# plt.plot(x, rf_reg.predict(K), color = 'yellow')






