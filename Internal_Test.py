# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:45:27 2018

@author: Sanmoy
"""
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from collections import Counter
import operator
from textblob import TextBlob as tb
#ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lmtzr=WordNetLemmatizer()
custom=set(stopwords.words('english')+list(punctuation)+[',', '“', '’'])

#################1. Summarization #################
text="The Indian state of Kerala has been devastated by severe floods. More than 350 people have died, while more than a million have been evacuated to over 4,000 relief camps. Tens of thousands remain stranded.The crisis is a timely reminder that climate change is expected to increase the frequency and magnitude of severe flooding across the world. Although no single flood can be linked directly to climate change, basic physics attests to the fact that a warmer world and atmosphere will hold more water, which will result in more intense and extreme rainfall.The monsoon season usually brings heavy rains but this year Kerala has seen 42% more rain than would be expected, with more than 2,300mm of rain across the region since the beginning of June, and over 700mm in August alone.These are similar levels seen during Hurricane Harvey, that hit Houston in August 2017, when more than 1,500mm of rain fell during one storm. Tropical cyclones and hurricanes, such as Harvey, are expected to increase in strength by up to 10% with a 2℃ rise in global temperature. Under climate change the probability of such extreme rainfall is also predicted to grow by up to sixfold towards the end of the century. The rivers and drainage systems of Kerala have been unable to cope with such large volumes of water and this has resulted in flash flooding.Much of that water would normally be slowed down by trees or other natural obstacles. Yet over the past 40 years Kerala has lost nearly half its forest cover, an area of 9,000 km², just under the size of Greater London, while the state’s urban areas keep growing. This means that less rainfall is being intercepted, and more water is rapidly running into overflowing streams and rivers."
words=[lmtzr.lemmatize(word.lower()) for word in word_tokenize(text) if word not in custom]
stemmed_para=" ".join([lmtzr.lemmatize(word.lower()) for word in word_tokenize(text) if word not in stopwords.words('english')])
original_sent=sent_tokenize(text)

counts=Counter(words)
count_common=counts.most_common()
final_words=[x for x in count_common if x[1]>2]
sents = sent_tokenize(stemmed_para)


d=dict()
for s in sents:
    score=0
    for w in final_words:
        if w[0] in s:
          score=score+w[1] 
    d[s]=score
 
#values=[]
#for key in d:
#    values.append(d[key])
#    
#values=sorted(values, key=int, reverse=True)
sorted_d=sorted(d.items(), key=operator.itemgetter(1), reverse=True)
summary=[]
for x in sorted_d[:3]:
    summary.append(x[0])
summary=''.join(summary)
print(summary)

################ 2. Extract the places############
import nltk
tokenized_text=word_tokenize(text)
tagged_sent=nltk.pos_tag(tokenized_text)
chunk_sent=nltk.ne_chunk(tagged_sent)
named_entities=[]
for tagged_tree in chunk_sent:
    if hasattr(tagged_tree, 'label'):
        print(tagged_tree)
        entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
        entity_type = tagged_tree.label()
        named_entities.append((entity_name, entity_type))

print(named_entities)
locations=[]
for loc in named_entities:
    if loc[1]=='GPE':
        locations.append(loc[0])
print(set(locations))


import spacy
sp = spacy.load('en', parser=False, matcher=False)
doc = sp(text)
places=[]
for ent in doc.ents:
    print(ent.text, ent.label_)
    if ent.label_=='GPE':
        places.append(ent.text)
print(set(places))



#################3. Polarity##################
blob = tb(text)
print(blob.polarity)
print(blob.sentiment)

################4. Get Top 5 words #####################
from gensim.corpora.dictionary import Dictionary
words_top=[lmtzr.lemmatize(word.lower()) for word in word_tokenize(text) if word.isalpha() and word not in custom]
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


from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)
acc=classifier.accuracy(testing)
print(acc)
classifier.classify('the weather is terrible')

######################## OR ########################
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
text_gram="Batman is a fictional superhero appearing in American comic books published by DC Comics."
bigram = list(ngrams((x.lower() for x in nltk.word_tokenize(text_gram) if x not in list(punctuation)) , 2)) 
print(bigram)
print(len(bigram))


###################### Ans-8 ###############
from nltk import pos_tag
text_2="I am planning to visit New Delhi to attend Analytics Vidhya Delhi Hackathon"
noun=[token for token, pos in pos_tag(word_tokenize(text_2)) if pos.startswith('N')]      
print(noun)
verb=[token for token, pos in pos_tag(word_tokenize(text_2)) if pos.startswith('V')]      
print(verb)
words_2=[word for word in word_tokenize(text_2) if word not in custom]
count_words2=Counter(words_2)
for w in count_words2:
    if count_words2[w]>1:
        print(w)
        
        
from textblob import TextBlob as tb
blob=tb(text_2)
a=blob.tags
for i,x in enumerate(a):
   if a[i][1]=='NNP':
       print(a[i])
       
       
       
###################### Create a World Cloud ###################
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(relative_scaling = 1.0, stopwords=set(STOPWORDS)).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
