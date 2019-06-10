from sklearn import preprocessing
import numpy as np
import re
text_data=["    Interrobang.  By Aishwarya Henriette    ",
           "Parking and going. By Karl Gautier",
           "    Today is the night.By Jarek Prakash   "]
strip_whitespace=[string.strip() for string in text_data]
remove_periods=[string.replace(".","") for string in strip_whitespace]
def capitalizer(string):
    return string.upper()
cap_text=[capitalizer(X) for X in remove_periods]
def replace_letters_with_X(string: str)->str:
    return re.sub(r"[a-zA-Z]","X",string)
replace_X=[replace_letters_with_X(string) for string in cap_text]
print(replace_X)

#移除标点
import unicodedata
import sys
text_data=['HI!!!I. Love. This. Song....',
           '1000% Agree!!!! #LoveIT',
           'Right?!?!']
punctuation=dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
remove_pun=[string.translate(punctuation) for string in text_data]
print(remove_pun)

#分词
#import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
string="the science of today is the technology of tomorrow"
string2="The science of today is the technology of tomorrow. Tomorrow is today."
print(word_tokenize(string))
print(sent_tokenize(string2))

#remove 停止词
from nltk.corpus import stopwords
#import nltk
#nltk.download('stopwords')
tokenized_words=['i','am','going','to','go','to','the','store','and','the','park']
stop_words=stopwords.words('english')
remove_stopwords=[word for word in tokenized_words if word not in stop_words]
print(remove_stopwords)

#词根
from nltk.stem.porter import PorterStemmer
tokenized_words=['i','am','humled','by','this','traditional','meeting']
porter=PorterStemmer()
tk_words=[porter.stem(word) for word in tokenized_words]
print(tk_words)

#词性
#import nltk
#nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
text_data="Chris and I loved outdoor running and playing"
text_tagged=pos_tag(word_tokenize(text_data))
print(text_tagged)

#训练自己的tagger
#import nltk
#nltk.download('brown')
from nltk.corpus import brown
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger
sentences=brown.tagged_sents(categories='news')
print(sentences)
train=sentences[:4000]
test=sentences[4000:]
unigram=UnigramTagger(train)
bigram=BigramTagger(train,backoff=unigram)
trigram=TrigramTagger(train,backoff=bigram)
print(trigram.evaluate(test))

#词袋
from sklearn.feature_extraction.text import CountVectorizer
text_data=np.array(['I love Brazil. Brazil!',
                    'Sweden is best',
                    'Germany beats both'])
count=CountVectorizer()
bag_of_words=count.fit_transform(text_data)
print(bag_of_words.toarray())

count_2gram=CountVectorizer(ngram_range=(1,2),stop_words="english",vocabulary=['brazil','love'])
bag=count_2gram.fit_transform(text_data)
print(bag.toarray())

#词的权重
#tf-idf
#tf-idf(t,d)=tf(t,d)*idf(t)
#tf 词在文档中出现的次数
#idf(t)=log((1+nd)/1+df(d,t))+1 nd表示文档数，df表示t出现的文档数
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
feature_matrix=tfidf.fit_transform(text_data)
print(feature_matrix.toarray())
