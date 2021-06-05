# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:56:53 2020

@author: Kutay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 - Veri Önişleme

# 2.1 - Veri Yükleme
veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values #bağımsız değişken

#K-Means++ Kümeleme
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4, init='k-means++')
Y_tahmin_knn=kmeans.fit_predict(X)
# print(kmeans.cluster_centers_)
plt.scatter(X[Y_tahmin_knn==0,0],X[Y_tahmin_knn==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin_knn==1,0],X[Y_tahmin_knn==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin_knn==2,0],X[Y_tahmin_knn==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin_knn==3,0],X[Y_tahmin_knn==3,1],s=100, c='yellow')
plt.title('K_Means')
plt.show()

sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i,init = 'k-means++' , random_state = 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    

plt.plot(range(1,10) , sonuclar)
plt.show()

#Hiyerarsik Kümeleme

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)
sonuc_2 = []

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()


import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()










