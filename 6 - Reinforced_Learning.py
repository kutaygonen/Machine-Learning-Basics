# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 17:26:42 2020

@author: Kutay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#RANDOM SELECTION

import random

N = 10000  #10k tıklama
d=10 #10 ilan  Ri(n)
toplam = 0
secilenler = []

for n in range(0,N):
    ad = random.randrange(d) # verilen n. satır = 1 ise odul = 1
    secilenler.append(ad)
    odul = veriler.values[n,ad]
    toplam = toplam+odul
    
# print(len(secilenler))
    
plt.hist(secilenler)
plt.show()
    
#Upper Confidence Bound - UCB
import math

oduller=[0]*d #Ni(n)
toplam = 0
toplam = toplam+odul
tiklamalar = [0]*d  #o ana kadar ki tıklamalar
secilenler_2 = []

for n in range(0,N):
    ad=0 #seçilen ilan
    max_ucb = 0
    for i in range(0,d):
            if(tiklamalar[i] > 0):
                ortalama = oduller[i]/tiklamalar[i]
                delta = math.sqrt((3/2)*math.log(n)/tiklamalar[i])
                ucb = ortalama+delta
            else:
                ucb = N*10            
            if max_ucb < ucb:
                max_ucb = ucb
                ad = i
    secilenler_2.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+1
    odul = veriler.values[n,ad]
    oduller[ad] = oduller[ad] + odul
    toplam = toplam+odul

print('Toplam Odul : \n' , toplam)
    
plt.hist(secilenler_2)
plt.show()


#Thampson Sampling

oduller=[0]*d #Ni(n)
toplam = 0
toplam = toplam+odul
secilenler_3 = []
birler = [0]*d
sifirlar=[0]*d

for n in range(1,N):
    ad=0 #seçilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate(birler[i]+1,sifirlar[i]+1)
        if(rasbeta > max_th):
            max_th = rasbeta
            ad = i
    secilenler_3.append(ad)
    odul = veriler.values[n,ad]
    if odul == 1:
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad]+1
    toplam = toplam+odul

print('Toplam Odul Thompson : \n' , toplam)
    
plt.hist(secilenler_3)
plt.show()





