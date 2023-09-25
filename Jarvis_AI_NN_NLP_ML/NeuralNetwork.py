import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import nltk.data
from nltk.tokenize import RegexpTokenizer

Stemmer = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')

def tokenize(sentence):
    return tokenizer.tokenize(sentence)


def stem(word):
    return Stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words),dtype =np.float32)
    
    for idx,w in enumerate(words):
        if w in sentence_word:
            bag[idx] = 1
        
    return bag