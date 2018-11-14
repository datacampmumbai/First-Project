# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:54:03 2018

@author: Sanmoy
"""
import nltk
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob as tb

sent = "Ever since SEBI mandated mutual fund houses to disclose how much fund managers, directors of the asset management company and other key employees invest their own money into their schemes, the data has become an important parameter to rate overall stewardship of funds."
sent
words=word_tokenize(sent)
words

blob = tb(sent)
blob
blob.words
words_blob = list(blob.words)
print(words_blob)

para="In the realm of Sentiment analysis, the main goal would be to classify the polarity of a given text at different levels —whether the expressed opinion in a document, a sentence or an entity feature/aspect is positive, negative, or neutral.For achieving this goal (polarity classification), you can see the whole process as a pipeline including different stages that can contribute to the accuracy of ending results. Subjectivity/objectivity identification can be seen as one of those stages that is commonly defined as classifying a given text (usually a sentence) into one of two classes: objective or subjective. For example, some researches showed that removing objective sentences from a document before classifying its polarity helped improve performance.Note that there are unique challenges for subjectivity detection: The subjectivity of words and phrases may depend on their context and an objective document may contain subjective sentences (e.g., a news article quoting people's opinions)."
para
blob = tb(para)
blob.sentences
type(blob.words)
blob_wor = list(blob.words)
blob_wor
blob.sentiment
blob.tags
print(list(blob.noun_phrases))

opinion = "I like Iphone, it's a pretty great phone"
blob = tb(opinion)
blob.sentiment

para_2="In late July 2018, severe flooding affected the south Indian state of Kerala due to unusually high rainfall during the monsoon season. It was the worst flooding in Kerala in nearly a century.[2] Over 322 people died, 15 are missing[3] within a fortnight, while at least a million [4][5] people were evacuated, mainly from Chengannur,[6] Pandanad,[7] Aranmula, Aluva, Chalakudy, Kuttanad and Pandalam. All 14 districts of the state were placed on red alert.[8][9] According to the Kerala government, one-sixth of the total population of Kerala had been directly affected by the floods and related incidents.[10] The Union government had declared it a Level 3 Calamity or 'Calamity of a severe nature"
blob = tb(para_2)
blob.noun_phrases
blob.sentiment


from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)

from nltk.book import *
text1.concordance("monstrous")
text2.concordance("monstrous")

text4.dispersion_plot(["citizens", "democracy","duties","America","freedom"])



####################### TF-IDF ####################################

articles = ["this car got the excellence award",\
         "good car gives good mileage",\
         "this car is very expensive",\
         "the company is growing with very high production",\
         "this company is financially good",
         "this company needs good management",
         "this company is reeling under losses"]
articles

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
vocabulary = set()
for doc in articles:
    print(doc)
    vocabulary.update(doc.split())

vocabulary
type(vocabulary)
vocabulary = list(vocabulary)
vocabulary
tfidf = TfidfVectorizer(vocabulary=vocabulary)
tfidf
tfidf.fit(articles)
tfidf.transform(articles)
import operator
final={}
for doc in articles:
    score={}
    X=tfidf.transform([doc])
    #print(X)
    for word in doc.split():
        score[word]=X[0, tfidf.vocabulary_[word]]
    scoredscore=sorted(score.items(), key=operator.itemgetter(1), reverse=True)
    print("\t", scoredscore)
    
    
    
spMat = tfidf.transform(articles)
from sklearn.metrics.pairwise import cosine_similarity
search_query = ["I owe a car"]
query_vector = tfidf.transform(search_query)
query_vector.toarray()
cosine_list = []
for doc_vec in spMat:
    x=cosine_similarity(query_vector, doc_vec)
    cosine_list.append(x[0][0])
cosine_list   
cosine_scores=sorted(cosine_list, reverse=True)
cosine_scores
    