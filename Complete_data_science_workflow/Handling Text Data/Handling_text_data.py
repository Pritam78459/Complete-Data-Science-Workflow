# -*- coding: utf-8 -*-

"""
Created on mon Jul 13 18:04:21 2020

@author: Pritam
"""

import re
from bs4 import BeautifulSoup
import unicodedata
import sys
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Cleaning Text

text_data = [" Interrobang. By Aishwarya Henriette ",
             "Parking And Going. By Karl Gautier",
             " Today Is The night. By Jarek Prakash "]

strip_whitespace = [string.strip() for string in text_data]

#print(strip_whitespace)

remove_periods = [string.replace(".", "") for string in strip_whitespace]

#print(remove_periods)

#String Capitalizer

def capitalizer(string: str) -> str:
    return string.upper()

#print([capitalizer(string) for string in remove_periods])
    
def replace_letters_with_X(string: str) -> str:
    return re.sub(r'[a-zA-Z]','X',string)

#print([replace_letters_with_X(string) for string in remove_periods])

#Parsing and cleaning HTML
    
html = """
    <div class='full_name'><span style='font-weight:bold'>
    Masego</span> Azra</div>"
    """
    
soup = BeautifulSoup(html,'lxml')

#print(soup.find("div", { "class" : "full_name" }).text)

#Removing Punctuations

text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

#print([string.translate(punctuation) for string in text_data])

#Tokenizing Text

string = 'The science of today is the technology of tomorrow'

#print(word_tokenize(string))

string = "The science of today is the technology of tomorrow. Tomorrow is today."

#print(sent_tokenize(string))

#Removing Stop Words

tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',    
                   'and',
                   'park']

stop_words = stopwords.words('english')

#print([word for word in tokenized_words if word not in stop_words])

tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

#Stemming Words

porter = PorterStemmer()

#print([porter.stem(word) for word in tokenized_words])

#Tagging parts of speech

text_data = "Chris loved outdoor running"

#text_tagged = pos_tag(word_tokenize(text_data))

#print(text_tagged)

#Encoding Text as bag of Words

text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])


count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

#print(bag_of_words)
#print(bag_of_words.toarray())

#print(count.get_feature_names())

count_2gram = CountVectorizer(ngram_range=(1,2),
                              stop_words="english",
                              vocabulary=['brazil'])

bag = count_2gram.fit_transform(text_data)
# View feature matrix
#print(bag.toarray())

# View the 1-grams and 2-grams
#print(count_2gram.vocabulary_)

#Weighting Word Importance

text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

#print(feature_matrix)

#print(feature_matrix.toarray())
#print(tfidf.vocabulary_)

