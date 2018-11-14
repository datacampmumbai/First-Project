# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:04:37 2018

@author: Sanmoy
"""

import time
import os
import pandas as pd
import numpy as np
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob as tb
import gensim
custom=set(stopwords.words('english')+list(punctuation))

import gzip
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    hdlr = logging.FileHandler('reviews.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)
    
##Set directory
path="C:/F/NMIMS/DataScience/Sem-3/TA/data/reviews_data"
os.chdir(path)



def read_input(file):
    logger.info("in read_input=========")
    try:
        with gzip.open(file, 'rb') as f:
            for i, line in enumerate(f):
                if(i%10000 == 0):
                    logger.info("read {} reviews".format(i))
                yield gensim.utils.simple_preprocess(line)
                
    except Exception as ex:
        logger.error(ex)
                
            
documents =list(read_input("reviews_data.txt.gz"))
logger.info("Done")
model = gensim.models.Word2Vec(documents, size=150, min_count=0, workers=10)            
model.train(documents, total_examples=len(documents), epochs=10)

model.wv.most_similar(positive=["cute"], topn=6)
model.wv.most_similar(negative=["cute"], topn=6)



from langdetect import detect
detect("मैं एक अच्छा लड़का हूँ")

detect("আমি একটা ভাল ছেলে")

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


text = "hi"
wordcloud = WordCloud(relative_scaling = 1.0, stopwords=set(STOPWORDS)).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
