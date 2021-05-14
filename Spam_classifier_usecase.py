# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:19:57 2021

@author: SOHAM KULKARNI
"""
"""SPAM CLASSIFIER"""

#importing library to load the data
import numpy as np
import pandas as pd

messages = pd.read_csv('/Users/SOHAM KULKARNI/OneDrive/Desktop/Practice/NLP/SpamClassification_usecase/SMSSpamCollection', sep = '\t', names = ['label', 'name'])
messages.head()

#Data cleaning and pre-processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
Wordnet = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-z]', ' ', messages['name'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
 
    
#Creating a bag of words model (Vectors)
from sklearn.feature_extraction.text import CountVectorizer #Create vectors of all the message
cv = CountVectorizer(max_features=2500) #Selecting 2500 records from the corpus
X = cv.fit_transform(corpus).toarray()

#Creating dummy variables for the label
y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


#Train-Test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Training the model using Naive-Bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect.predict(X_test)

"""Evaluation metrics"""
#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
 



