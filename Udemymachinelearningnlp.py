
# coding: utf-8

# In[4]:


import nltk


# In[5]:


nltk.download()


# In[6]:


from nltk.book import *


# In[7]:


# concordance


# In[8]:


text1.concordance("monstrous")


# In[9]:


text2.concordance('monstrous')


# In[10]:


text2.similar("monstrous")


# In[11]:


text2.common_contexts(['monstrous'])


# In[13]:


import matplotlib as p


# In[16]:


text4.dispersion_plot(['citizens','president','freedom'])


# In[17]:


from nltk.tokenize import word_tokenize, sent_tokenize


# In[18]:


text= 'The era has faced with explosive growth in data generation. Data generation has undergone a renaissance change. This availability of data has led a paradigm shift in the E-commerce sector; data is no longer a by-product of business activities, but are the asset to a company it helps in providing insights which are required in satisfying customers’ needs. This paper provides an overview of Sentiment Analysis of product reviews based on different algorithms and its efficiency in determining positive from negative reviews based on N-gram, Bigram with the application of Count-Vectorizer and TFIDF Matrix.'


# In[21]:


sents= sent_tokenize(text)
print(sents)
len(sents)


# In[22]:


word= word_tokenize(text)


# In[24]:


print((word))


# In[25]:


from nltk.corpus import stopwords


# In[26]:


from string import punctuation


# In[27]:


cusomstopwors= (stopwords.words('english')+list(punctuation))


# In[32]:


stopword= [word for word in word_tokenize(text) if word not in cusomstopwors]


# In[33]:


print(stopword)


# In[39]:


from nltk.stem.lancaster import LancasterStemmer
st= LancasterStemmer()
text2= 'Different classification models have been employed to check the prediction accuracy of the unlabeled text. Based on the above classification and tool has been developed which predicts the incoming reviews and classify its sentiment polarity.  '



# In[40]:


stemedword= [st.stem(word) for word in word_tokenize(text2)]


# In[41]:


print(stemedword)


# In[42]:


# noun verbs conjustion


# In[43]:


nltk.pos_tag(word_tokenize(text2))


# In[50]:


from bs4 import BeautifulSoup


# In[55]:


html=['<html><heading style="font-size:20px"><i>This is the title<br><br></i></heading>',
     '<body><b>This is the body</b><p id="para1">This is para1<a href="www.google.com">Google</a></p>',
     '<p id="para2">This is para 2</p></body></html>']


# In[56]:


html=''.join(html)


# In[59]:


soup= BeautifulSoup(html)
print(soup.prettify())


# In[61]:


soup.html.name


# In[62]:


soup.html.body


# In[63]:


soup.body.name


# In[64]:


soup.body.text


# In[65]:


soup.html.contents


# In[66]:


soup.body.parent.name


# In[67]:


soup.b.next_sibling


# In[72]:


bold= soup.findAll("b")
print(bold)


# In[75]:


print(bold[0].text)


# In[76]:


paras= ''.join([p.text for p in soup.findAll('p')])
print(paras)


# In[77]:


soup.findAll(id="para2")[0].text


# In[90]:


font20=' '.join([p.text for p in soup.findAll(style="font-size:20px")])


# In[91]:


print(font20)


# In[92]:


soup.findAll(['b','p'])


# In[93]:


soup.findAll({'b':True, 'p':'False'})


# In[96]:


link= soup.find('a')
print(link)


# In[100]:


print(link['href']+' is the url and' + 'link.text'+ 'is the text')


# In[103]:


soup.find(text= 'Google').findNext('p').text


# In[110]:


import urllib


# In[140]:


def getNYYPostText(url, token):
       response= requests.get(url)
       soup= BeautifulSoup(response.content)
       page= str(soup)
       title= soup.find('title').text
       divs=soup.find_all('p',{'class':'css-4w7y5l'})
       text=''.join(map(lambda p:p.text, divs))
       
       return text, title
       

       
       


# In[141]:


someUrl = "https://www.nytimes.com/2018/11/12/world/middleeast/jamal-khashoggi-killing-saudi-arabia.html"
# the article we would like to summarize
textOfUrl = get_only_text_washington_post_url(someUrl)
# get the title and text
fs = FrequencySummarizer()
# instantiate our FrequencySummarizer class and get an object of this class
summary = fs.summarize(textOfUrl[1], 3)
# get a summary of this article that is 3 sentences long


# In[142]:


summary


# In[136]:


from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest


# In[139]:





# In[6]:


######################################################################################
# THis example is pretty much entirely based on this excellent blog post
# http://glowingpython.blogspot.in/2014/09/text-summarization-with-nltk.html
# Thanks to TheGlowingPython, the good soul that wrote this excellent article!
# That blog is is really interesting btw.
######################################################################################


######################################################################################
# nltk - "natural language toolkit" is a python library with support for 
#         natural language processing. Super-handy.
# Specifically, we will use 2 functions from nltk
#  sent_tokenize: given a group of text, tokenize (split) it into sentences
#  word_tokenize: given a group of text, tokenize (split) it into words
#  stopwords.words('english') to find and ignored very common words ('I', 'the',...) 
#  to use stopwords, you need to have run nltk.download() first - one-off setup
######################################################################################
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords

######################################################################################
# We have use dictionaries so far, but now that we have covered classes - this is a good
# time to introduce defaultdict. THis is class that inherits from dictionary, but has
# one additional nice feature: Usually, a Python dictionary throws a KeyError if you try 
# to get an item with a key that is not currently in the dictionary. 
# The defaultdict in contrast will simply create any items that you try to access 
# (provided of course they do not exist yet). To create such a "default" item, it relies 
# a function that is passed in..more below. 
######################################################################################
from collections import defaultdict

######################################################################################
#  punctuation to ignore punctuation symbols
######################################################################################
from string import punctuation

######################################################################################
# heapq.nlargest is a function that given a list, easily and quickly returns
# the 'n' largest elements in the list. More below
######################################################################################
from heapq import nlargest


######################################################################################
# Our first class, named FrequencySummarizer 
######################################################################################
class FrequencySummarizer:
    # indentation changes - we are now inside the class definition
    def __init__(self, min_cut=0.1, max_cut=0.9):
        # The constructor named __init__
        # THis function will be called each time an object of this class is 
        # instantiated
        # btw, note how the special keyword 'self' is passed in as the first
        # argument to each method (member function).
        self._min_cut = min_cut
        self._max_cut = max_cut 
        # Words that have a frequency term lower than min_cut 
        # or higer than max_cut will be ignored.
        self._stopwords = set(stopwords.words('english') + list(punctuation))
        # Punctuation symbols and stopwords (common words like 'an','the' etc) are ignored
        #
        # Here self._min_cut, self._max_cut and self._stopwords are all member variables
        # i.e. each object (instance) of this class will have an independent version of these
        # variables. 
        # Note how this function is used to set up the member variables to their appropriate values
    # indentation changes - we are out of the constructor (member function, but we are still inside)
    # the class.
    # One important note: if you are used to programming in Java or C#: if you define a variable here
    # i.e. outside a member function but inside the class - it becomes a STATIC member variable
    # THis is an important difference from Java, C# (where all member variables would be defined here)
    # and is a common gotcha to be avoided.

    def _compute_frequencies(self, word_sent):
        # next method (member function) which takes in self (the special keyword for this same object)
        # as well as a list of sentences, and outputs a dictionary, where the keys are words, and
        # values are the frequencies of those words in the set of sentences
        freq = defaultdict(int)
        # defaultdict, which we referred to above - is a class that inherits from dictionary,
        # with one difference: Usually, a Python dictionary throws a KeyError if you try 
        # to get an item with a key that is not currently in the dictionary. 
        # The defaultdict in contrast will simply create any items that you try to access 
        # (provided of course they do not exist yet). THe 'int' passed in as argument tells
        # the defaultdict object to create a default value of 0
        for s in word_sent:
        # indentation changes - we are inside the for loop, for each sentence
          for word in s:
            # indentation changes again - this is an inner for loop, once per each word_sent
            # in that sentence
            if word not in self._stopwords:
                # if the word is in the member variable (dictionary) self._stopwords, then ignore it,
                # else increment the frequency. Had the dictionary freq been a regular dictionary (not a 
                # defaultdict, we would have had to first check whether this word is in the dict
                freq[word] += 1
        # Done with the frequency calculation - now go through our frequency list and do 2 things
        #   normalize the frequencies by dividing each by the highest frequency (this allows us to 
        #            always have frequencies between 0 and 1, which makes comparing them easy
        #   filter out frequencies that are too high or too low. A trick that yields better results.
        m = float(max(freq.values()))
        # get the highest frequency of any word in the list of words
        for w in freq.keys():
            # indentation changes - we are inside the for loop
            freq[w] = freq[w]/m
            # divide each frequency by that max value, so it is now between 0 and 1
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                # indentation changes - we are inside the if statement - if we are here the word is either
                # really common or really uncommon. In either case - delete it from our dictionary
                del freq[w]
                # remember that del can be used to remove a key-value pair from the dictionary
        return freq
        # return the frequency list

    def summarize(self, text, n):
        # next method (member function) which takes in self (the special keyword for this same object)
        # as well as the raw text, and the number of sentences we wish the summary to contain. Return the 
        # summary
        sents = sent_tokenize(text)
        # split the text into sentences
        assert n <= len(sents)
        # assert is a way of making sure a condition holds true, else an exception is thrown. Used to do 
        # sanity checks like making sure the summary is shorter than the original article.
        word_sent = [word_tokenize(s.lower()) for s in sents]
        # This 1 sentence does a lot: it converts each sentence to lower-case, then 
        # splits each sentence into words, then takes all of those lists (1 per sentence)
        # and mushes them into 1 big list
        self._freq = self._compute_frequencies(word_sent)
        # make a call to the method (member function) _compute_frequencies, and places that in
        # the member variable _freq. 
        ranking = defaultdict(int)
        # create an empty dictionary (of the superior defaultdict variety) to hold the rankings of the 
            # sentences. 
        for i,sent in enumerate(word_sent):
            # Indentation changes - we are inside the for loop. Oh! and this is a different type of for loop
            # A new built-in function, enumerate(), will make certain loops a bit clearer. enumerate(sequence), 
            # will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth.
            # A common idiom to change every element of a list looks like this:
            #  for i in range(len(L)):
            #    item = L[i]
            #    ... compute some result based on item ...
            #    L[i] = result
            # This can be rewritten using enumerate() as:
            # for i, item in enumerate(L):
            #    ... compute some result based on item ...
            #    L[i] = result
            for w in sent:
                # for each word in this sentence
                if w in self._freq:
                    # if this is not a stopword (common word), add the frequency of that word 
                    # to the weightage assigned to that sentence 
                    ranking[i] += self._freq[w]
        # OK - we are outside the for loop and now have rankings for all the sentences
        sents_idx = nlargest(n, ranking, key=ranking.get)
        # we want to return the first n sentences with highest ranking, use the nlargest function to do so
        # this function needs to know how to get the list of values to rank, so give it a function - simply the 
        # get method of the dictionary
        return [sents[j] for j in sents_idx]
       # return a list with these values in a list
# Indentation changes - we are done with our FrequencySummarizer class!


######################################################################################
# Now to get a URL and summarize
######################################################################################


# In[ ]:





# In[129]:


import urllib2
from bs4 import BeautifulSoup

######################################################################################
# Introducing Beautiful Soup: " Beautiful Soup is a Python library for pulling data out of 
# HTML and XML files. It works with your favorite parser to provide idiomatic ways of 
# navigating, searching, and modifying the parse tree. It commonly saves programmers hours 
# or days of work.
######################################################################################



def get_only_text_washington_post_url(url):
    # This function takes in a URL as an argument, and returns only the text of the article in that URL. 
    page = urllib2.urlopen(url).read().decode('utf8')
    # download the URL
    soup = BeautifulSoup(page)
    # initialise a BeautifulSoup object with the text of that URL
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    # use this code to get everything in that text that lies between a pair of 
    # <article> and </article> tags. We do this because we know that the URLs we are currently
    # interested in - those from the WashingtonPost have this nice property

    # OK - we got everything between the <article> and </article> tags, but that everything
    # includes a bunch of other stuff we don't want
    # Now - repeat, but this time we will only take what lies between <p> and </p> tags
    # these are HTML tags for "paragraph" i.e. this should give us the actual text of the article
    soup2 = BeautifulSoup(text)
    if soup2.find_all('p')!=[]:
        text = ' '.join(map(lambda p: p.text, soup2.find_all('p')))
    # use this code to get everything in that text that lies between a pair of 
    # <p> and </p> tags. We do this because we know that the URLs we are currently
    # interested in - those from the WashingtonPost have this nice property
    return soup.title.text, text
# Return a pair of values (article title, article body)
# Btw note that BeautifulSoup return the title without our doing anything special - 
# this is why BeautifulSoup works so much better than say regular expressions at parsing HTML


#####################################################################################
# OK! Now lets give this code a spin
#####################################################################################


# In[130]:


someUrl = "https://www.washingtonpost.com/news/the-switch/wp/2015/08/06/why-kids-are-meeting-more-strangers-online-than-ever-before/"
# the article we would like to summarize
textOfUrl = get_only_text_washington_post_url(someUrl)
# get the title and text
fs = FrequencySummarizer()
# instantiate our FrequencySummarizer class and get an object of this class
summary = fs.summarize(textOfUrl[1], 3)
# get a summary of this article that is 3 sentences long


# In[121]:


summary


# In[143]:


fs = FrequencySummarizer()
# instantiate our FrequencySummarizer class and get an object of this class
summary = fs.summarize(textOfUrl[1], 3)
# get a summary of this article that is 3 sentences long


# In[144]:


someUrl


# In[163]:


import requests
import urllib2
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# In[164]:


def getwashpost(url,token):
    try:
        page= urllib2.urlopen(url).read().decode('utf8')
    except:
        retrun(None,None)
        soup= BeautifulSoup(page)
        if soup is None:
            return(None,None)
        text= ''
        if soup.find_all(token) is not None:
            text= ''.join(map(lambda p: p.text, soup.find_all(token)))
            soup2= BeautifulSoup(text)
            if soup2.find_all('p')is  not None:
                text= ''.join(map(lambda p: p.text, soup2.find_all('p')))
        return text, soup.title.text    
    

