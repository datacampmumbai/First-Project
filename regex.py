# -*- coding: utf-8 -*-


import regex as re
article = ["big data", "metadata", "data science", "Bangalore", "teradataing"]       
lst=[]
pattern=".*data"
lst+=[a for a in article if re.search(pattern, a)]
print(lst)

article = ["bat", "mat", "cat", "rat", "dat", "tat", "fat", "mom"]
lst=[]
lst+=[a for a in article if re.match(r"[b,r]", a)]
print(lst)

lst=[]
lst+=[a for a in article if re.search(r"^[br]", a)]
print(lst)

lst=[]
lst+=[a for a in article if re.search(r"t$", a)]
print(lst)

article = ["bat", "robotics", "megabyte", "bangalore"]
lst=[]
pattern="b.t"
lst+=[a for a in article if re.search(pattern, a)]
print(lst)

article_1 = ["bat", "baat", "baet", "baeet", "btat", "ba123t", "batsman"]
article_2 = ["dat", "tat", "that", "that", "4m", "bat", "daaaat"]
pattern="^b.*t$"
for indx, word in enumerate(article_1):
    if re.search(pattern, word):
        article_1[indx]="bat"
print(article_1)

lst=[]
pattern="^[dt].*t$"
lst+=[a for a in article_2 if re.search(pattern, a)]
print(lst)

lst=[]
article = ["Warot 1812", "There are 5820 ducks", "Happy new year 2016", "2812 feet"]

article = ["Bangalore1234", "Chandigarh78", "B64ombay", "durgapur12", "kolkata"]
lst=[]
pattern="[A-z]\d+"
lst+=[re.sub("\d+","", a) for a in article if re.search(pattern, a)]
print(lst)

t="Bangalore1234"
t="".join(a for a in t if a.isalpha())
print(t)

article = ["[bat]", "mat", "cat", "[rat]", "rat","dat", "tat", "fat"]
lst=[]
lst+=[a for a in article if re.search("^\[[r]", a)]
print(lst)

article = ["[bat]", "mat", "cat", "[rat]", "rat","[dat]", "tat", "fat"]
lst=[]
lst+=[a for a in article if re.search("\[(.*)]", a)]
print(lst)

string="My height is 5.11 feet"
pattern="\d+(\.\d{1,2})?"
x=re.search(pattern, string)
print(x.group(0))

