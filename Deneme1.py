# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:54:01 2020

@author: Kutay
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 - Veri Önişleme

# 2.1 - Veri Yükleme
veriler = pd.read_csv('Biase.txt',delimiter="\t")
veriler = veriler.iloc[0:500,:]

X = veriler.iloc[:,1:58]


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(X)

 

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init='k-means++')
Y_tahmin=kmeans.fit_predict(X)
# print(kmeans.cluster_centers_)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('K_Means')
plt.show()

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)
sonuc_2 = []

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

from sklearn.model_selection import GridSearchCV

p = [{'n_clusters': range(10)}] #GSCV nin aramak istediği parametreler


gs = GridSearchCV(estimator = kmeans,param_grid = p, scoring='accuracy', cv = 10 ,n_jobs =-1)

grid_search = gs.fit(X,Y_tahmin)
en_iyi_sonuc = grid_search.best_score_
en_iyi_parametreler = grid_search.best_params_
