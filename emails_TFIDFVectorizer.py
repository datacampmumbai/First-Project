# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:34:35 2018

@author: Sanmoy
"""
import os
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
custom=set(stopwords.words('english')+list(punctuation))
path="C:/F/NMIMS/DataScience/Sem-3/TA/data/spam/enron1.tar/enron1/enron1/emails"
os.chdir(path)
features_set = []
labels = []
for email in os.listdir(os.getcwd()):
    words = []
    text = open(email, encoding="latin-1")
    words = " ".join([x for x in text.read().split(" ") if x.isalpha()])
    features_set.append(words)
    if "ham" in email:
        labels.append(1)
    else:
        labels.append(0)
        
        

from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
x_train, x_test, y_train, y_test = tts(features_set, labels, test_size=0.20, random_state=53)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
nb_classifier = MultinomialNB()
nb_classifier.fit(tfidf_train, y_train)
preds = nb_classifier.predict(tfidf_test)
acc_spam1 = metrics.accuracy_score(y_test, preds)
print(acc_spam1)
cm_spam = metrics.confusion_matrix(y_test, preds, labels=[0,1])
print(cm_spam)     