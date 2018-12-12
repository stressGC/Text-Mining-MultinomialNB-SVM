"""
utility class used to preprocess the data
"""

import string
import nltk
import re
from nltk.stem import WordNetLemmatizer

#load the stopword list from nltk library
stoplist = [word for word in nltk.corpus.stopwords.words('english')] 
wnl = WordNetLemmatizer()

def no_punctuation_unicode(text):
    '''.translate only takes str. Therefore, to use .translate in the 
    tokenizer in TfidfVectorizer I need to write a function that converts 
    unicode -> string, applies .translate, and then converts it back'''
    str_text = str(text)
    no_punctuation = str_text.translate(str.maketrans('','',string.punctuation))
    unicode_text = no_punctuation
    return unicode_text

def contains_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def prep_review(review):
    lower_case = review.lower()
    no_punct = no_punctuation_unicode(lower_case)
    tokens = nltk.word_tokenize(no_punct)    
    has_letters = [t for t in tokens if re.search('[a-zA-Z]', t)]
    no_small_words = [t for t in tokens if len(t) > 2]
    drop_numbers  = [t for t in no_small_words if not contains_numbers(t)]
    drop_stops = [t for t in drop_numbers if not t in stoplist] 
    lemmed = [wnl.lemmatize(word) for word in drop_stops]
    return lemmed
