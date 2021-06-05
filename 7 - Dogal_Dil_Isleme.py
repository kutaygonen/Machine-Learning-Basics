# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:56:23 2020

@author: Kutay
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv' , error_bad_lines=False)

#Eksik Veri
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan , strategy = 'most_frequent')

Y = yorumlar.iloc[:,-1:].values

imputer = imputer.fit(Y[:,-1:])
Y[:,-1:] = imputer.transform(Y[:,-1:])


#DATA PREPROCESSING
import re
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
 
from nltk.corpus import stopwords

derlem = []
for i in range(716):
    yorum = re.sub('[^a-zA-Z]',' ', yorumlar['Review'][i])
    
    yorum = yorum.lower()
    yorum = yorum.split()
    
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] #kelimenin gövedesinikökünü bul listeye çevir
    yorum = ' '.join(yorum)
    derlem.append(yorum)

#FEATURE EXTRACTION
#COUNT VECTORIZER
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1152)

X = cv.fit_transform(derlem).toarray()

#Machine Learning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=0)

# Gaussion NB

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)










