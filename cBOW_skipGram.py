# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:10:42 2018

@author: Sanmoy
"""
import os
import pandas as pd
import gensim
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
custom=set(stopwords.words('english')+list(punctuation)+['»'])

path="C:/F/NMIMS/DataScience/Sem-3/TA/data"
os.chdir(path)

data = open("11-0.txt", encoding="latin-1").read()
doc = data.replace("\n", " ")
print(doc)
data = []
for sent in sent_tokenize(doc):
    temp = []
    for j in word_tokenize(sent):
        if j not in custom:
            temp.append(j.lower())
    data.append(temp)
len(data)
data

##Create CBOW Model
model1 = gensim.models.Word2Vec(data, min_count=1, size=100, window=5)
print("Cosine similarity between 'alice' "+"and'wonderland'-CBOW: ", model1.similarity('alice', 'wonderland'))
print("Cosine similarity between 'alice' "+"and'machines'-CBOW: ", model1.similarity('alice', 'machines'))


from textblob import TextBlob as tb
blob = tb(doc)
blob_wor = list(blob.words)
blob_wor
data = [word.lower() for word in blob_wor if word not in custom]
model1 = gensim.models.Word2Vec([data], min_count=1, size=100, window=5)
print(model1.similarity('after', 'like'))

#data = [word for word in data if word not in custom]    
