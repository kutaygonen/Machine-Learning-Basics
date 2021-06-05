# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:19:06 2020

@author: Kutay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('GSM2230757_human1_umifm_counts.csv')
veriler = veriler.iloc[0:500,:]
y = veriler.iloc[:,2:3].values
x = veriler.iloc[:,3:20128].values

# 2.3.1 Label Encoder - #12 Cluster
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le= LabelEncoder()
y[:,0] = le.fit_transform(y[:,0])


from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x = pca.fit_transform(x)

# plt.scatter(x[:,:], range(500))
plt.scatter(y,x)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x = sc.fit_transform(x)




 


# 2.3.2 One Hot Encoder
# ohe=OneHotEncoder(categories='auto')
# cells=ohe.fit_transform(cells).toarray()


# K-Means++ Kümeleme
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters = 4, init='k-means++')
# Y_tahmin = kmeans.fit_predict(x)


# print(kmeans.cluster_centers_)
# sonuclar = []


# for i in range(1,13):
#     kmeans = KMeans(n_clusters = i,init = 'k-means++' , random_state = 123)
#     kmeans.fit(x)
#     sonuclar.append(kmeans.inertia_)
    
# plt.scatter(x[Y_tahmin==0,0],x[Y_tahmin==0,1],s=100, c='red')
# plt.scatter(x[Y_tahmin==1,0],x[Y_tahmin==1,1],s=100, c='blue')
# plt.scatter(x[Y_tahmin==2,0],x[Y_tahmin==2,1],s=100, c='green')
# plt.scatter(x[Y_tahmin==3,0],x[Y_tahmin==3,1],s=100, c='yellow')
# plt.title('K_Means')
# plt.show()


# plt.plot(range(1,13) , sonuclar)
# plt.show()



# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state=0)


# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)


#Hiyerarsik Kümeleme

# from sklearn.cluster import AgglomerativeClustering

# ac = AgglomerativeClustering(n_clusters=13, affinity='euclidean', linkage='ward')
# Y_tahmin = ac.fit_predict(x)
# print(Y_tahmin)



# plt.scatter(x[Y_tahmin==0,0],x[Y_tahmin==0,1],s=100, c='blue')
# plt.scatter(x[Y_tahmin==1,0],x[Y_tahmin==1,1],s=100, c='blue')
# plt.scatter(x[Y_tahmin==2,0],x[Y_tahmin==2,1],s=100, c='green')
# plt.scatter(x[Y_tahmin==3,0],x[Y_tahmin==3,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==4,0],x[Y_tahmin==4,1],s=100, c='black')
# plt.scatter(x[Y_tahmin==5,0],x[Y_tahmin==5,1],s=100, c='magenta')
# plt.scatter(x[Y_tahmin==6,0],x[Y_tahmin==6,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==7,0],x[Y_tahmin==7,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==8,0],x[Y_tahmin==8,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==9,0],x[Y_tahmin==9,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==10,0],x[Y_tahmin==10,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==11,0],x[Y_tahmin==11,1],s=100, c='yellow')
# plt.scatter(x[Y_tahmin==12,0],x[Y_tahmin==12,1],s=100, c='yellow')
# plt.show()

# plt.scatter(x,range(500))


# plt.title('HC')
# plt.show()


# import scipy.cluster.hierarchy as sch

# dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
# plt.show()









