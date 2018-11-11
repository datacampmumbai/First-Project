# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 08:56:16 2018

@author: Ashish
"""



import pandas as pd
import os 
path="C:/Users/Ashish/.spyder-py3/Swta"
os.chdir(path)

df = pd.read_csv("fake_or_real_news.csv")
df.info()
df.head(4)
df.shape()

from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer

y=df['label']

x_train,x_test,y_train,y_test = tts(df['text'],y,test_size=0.30, random_state=53)

len(x_test)

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(x_train)

count_test = count_vectorizer.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb_classifier = MultinomialNB()

nb_classifier.fit(count_train,y_train)

preds = nb_classifier.predict(count_test)

#calculate accuracy score

score = metrics.accuracy_score(y_test,preds)
print(score)

#Calculate confusion matrix
cm= metrics.confusion_matrix(y_test,preds,labels=['FAKE','REAL'])
print(cm)



import spacy 
article=df['text'][1]
#Instantiate the English model
sp = spacy.load('en', tagger=False, parser=False, matcher=False)

#Create a new document : doc
doc=sp(article)

#PRint all of the found entities 
for ent in doc.ents:
    print(ent.label,ent.text)

for token in doc:
    print(token.text,token.tag_)
    

asd = spacy.load('en', parser=False, matcher=False)
docs=asd(article)

#PRint all of the found entities 
for ent in doc.ents:
    print(ent.label_,ent.text)
    