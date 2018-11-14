# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:58:52 2018

@author: 
"""
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from collections import Counter
import operator
from textblob import TextBlob as tb
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lmtzr=WordNetLemmatizer()
custom=set(stopwords.words('english')+list(punctuation))


######################1. Text Summarisation #########################################
text="Around a dozen bags are up for sale on the website eBay, with asking prices of up to €1,000. The bags were given to the 1,200 members of the public chosen by ballot to follow proceedings from the grounds of Windsor Castle. The bags contain items such as orders of service, fridge magnets, ponchos, shortbread and chocolate coins. One online listing says it offers the chance to buy your very own piece of British royal history Friday's wedding was also attended by 850 private guests, who were not given gift bags. Among those 850 were celebrities such as model Cara Delevingne and singer Robbie Williams. Gift bags were also given to the 2,640 members of the public invited into the grounds of Windsor Castle for the Duke and Duchess of Sussex's wedding in May. A number of those gift bags were also sold online, with many auctions fetching more than €1,000."
print(text)
stemmed_para=" ".join([lmtzr.lemmatize(word) for word in word_tokenize(text)])
stemmed_words=[ps.stem(word.lower()) for word in word_tokenize(text)]

#original_sent=sent_tokenize(text)

counts=Counter(stemmed_words)
count_common=counts.most_common()
final_words=[x for x in count_common if x[1]>1]
sents = sent_tokenize(stemmed_para)
d=dict()
for s in sents:
    score=0
    for w in final_words:
        if w[0] in s:
          score=score+w[1] 
    d[s]=score
sorted_d=sorted(d.items(), key=operator.itemgetter(1), reverse=True)
summary=[]
for x in sorted_d[:3]:
    summary.append(x[0])
summary=''.join(summary)
print(summary)



##############2. Extract only the places using NER #########################
import nltk
tokenized_text=word_tokenize(text)
tagged_sent=nltk.pos_tag(tokenized_text)
chunk_sent=nltk.ne_chunk(tagged_sent)
named_entities=[]
for tagged_tree in chunk_sent:
    if hasattr(tagged_tree, 'label'):
        #print(tagged_tree)
        entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
        entity_type = tagged_tree.label()
        named_entities.append((entity_name, entity_type))

print(named_entities)
locations=[]
for loc in named_entities:
    if loc[1]=='GPE':
        locations.append(loc[0])
print(set(locations))



#import spacy
#sp = spacy.load('en', parser=False, matcher=False)
#doc = sp(text)
#places=[]
#for ent in doc.ents:
#    print(ent.label_, ent.text)
#    if ent.label_=='GPE':
#        places.append(ent.text)
#print(set(places))

############# Polarity and Subjectivity of para #########################
blob = tb(text)
print(blob.polarity)
print(blob.sentiment)


################4. Get Top 5 words #####################
custom_2=set(stopwords.words('english')+list(punctuation))
from gensim.corpora.dictionary import Dictionary
words_top=[lmtzr.lemmatize(word.lower()) for word in word_tokenize(text) if word.isalpha() and word not in custom_2]
dictnry = Dictionary([words_top])
corp = [dictnry.doc2bow(article) for article in [words_top]]
text_doc=corp[0]
sorted_doc = sorted(text_doc, key = lambda w:w[1], reverse=True)
for word_id, word_cnt in sorted_doc[:5]:
    print(dictnry.get(word_id), word_cnt)
        

#####OR#####
count_words=Counter(words_top)
top_count=count_words.most_common(5)
print(top_count)

############# Topics ##############
import gensim
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(corp, num_topics=3, id2word = dictnry, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))


############################5. #####################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
custom_1=set(stopwords.words('english')+list(punctuation))
training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')]


df_train=pd.DataFrame(training, columns=['text', 'label'])
for idx, eachRow in df_train.iterrows():
    newRow=" ".join([word for word in word_tokenize(eachRow[0]) if word not in custom_1])
    eachRow[0]=newRow

df_test=pd.DataFrame(testing, columns=['text', 'label'])
for idx, eachRow in df_test.iterrows():
    newRow=" ".join([word for word in word_tokenize(eachRow[0]) if word not in custom_1])
    eachRow[0]=newRow

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(df_train['text'])
y_train = df_train['label'].replace({'pos':0, 'neg':1})

count_test = count_vectorizer.transform(df_test['text'])
y_test = df_test['label'].replace({'pos':0, 'neg':1})

nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)   
           
y_pred = nb_classifier.predict(count_test)
acc_test = metrics.accuracy_score(y_test, y_pred)
print("Test Set accuracy for NB: {0}".format(acc_test))


test_text="the weather is terrible"
test_pre=" ".join([word for word in word_tokenize(test_text) if word not in custom_1])
pred_df=pd.DataFrame({'text' : test_pre}, index=[0])
count_pred=count_vectorizer.transform(pred_df['text'])
pred = nb_classifier.predict(count_pred)
pred[0]
if pred[0]==0:
    print('pos')
else:
    print('neg')
    
   
    
################6. Parse a News Article ###############
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
http = urllib3.PoolManager()
response = http.request('GET',"https://economictimes.indiatimes.com/news/economy/agriculture/indigo-launches-grow-indigo-a-joint-venture-with-mahyco-grow/articleshow/66360487.cms")
soup = BeautifulSoup(response.data, 'html.parser')
text2 = ". ".join([p.text for p in soup.find_all('div', {'class':'Normal'})])
print(text2)



############7. Bigrams ###########
from nltk.util import ngrams
text_gram="Analytics Vidhya is a great source to learn data science."
bigram = list(ngrams((x.lower() for x in word_tokenize(text_gram) if x not in list(punctuation)) , 2)) 
print(bigram)
print(len(bigram))



###################### Ans-8 ###############
from nltk import pos_tag
text_2="I am planning to visit New Delhi to attend Analytics Vidhya Delhi Hackathon"
noun=[token for token, pos in pos_tag(word_tokenize(text_2)) if pos.startswith('N') or pos.startswith('P')]      
print(noun)
print("No of words with noun as parts of speech tag: {0}".format(len(noun)))
verb=[token for token, pos in pos_tag(word_tokenize(text_2)) if pos.startswith('V')]      
print(verb)
print("No of words with verb as parts of speech tag: {0}".format(len(verb)))
words_2=[word for word in word_tokenize(text_2)]
count_words2=Counter(words_2)
freq=[]
for w in count_words2:
    if count_words2[w]>1:
        print(w)
        freq.append(w)
        
print("No of words with frequency count greater than one: {0}".format(len(freq)))
  
for token, pos in pos_tag(word_tokenize(text_2)):
    print(token, pos)


from textblob import TextBlob as tb
blob=tb(text_2)
blob.tags
for i,x in enumerate(a):
   if a[i][1]=='NNP':
       print(a[i])  


