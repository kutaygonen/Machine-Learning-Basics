# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:53:12 2020

@author: Kutay
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 - Veri Önişleme

# 2.1 - Veri Yükleme
veriler = pd.read_csv('eksikveriler.csv')

# 2.2 - Eksik Veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan , strategy = 'mean')

Yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
# print(Yas)  

# 2.3 - Encoder - Kategorik -> Numerik

ulke = veriler.iloc[:,0:1].values
# print(ulke)

# 2.3.1 Label Encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le= LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
# print(ulke)

# 2.3.2 One Hot Encoder
ohe=OneHotEncoder(categories='auto')
ulke=ohe.fit_transform(ulke).toarray()
# print(ulke)


# 2.4 - Numpy Dizilerinden Data Frame oluşturma
sonuc = pd.DataFrame(data = ulke, index = range(22) , columns = ['fr','tr' , 'us'])
# print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index=range(22), columns = ['Boy', 'kilo', 'yaş'])
# print(sonuc2)

sonuc3 = veriler.iloc[:,-1].values
sonuc3 = pd.DataFrame(data = sonuc3, index=range(22), columns = ['cinsiyet'])
# print(sonuc3)

# 2.4.1 - Data Frame Birleştirme - Concat ile
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)


s2=pd.concat([s,sonuc3] , axis=1)
print(s2)

# 2.5 - Veri bölmesi-Test ve Train(Eğitim)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state=0)

# 2.6 - Verilerin Ölçeklendirmesi - Standardization

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)




