# domain_name.train_data

import pymystem3

from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

import numpy as np
import re


def convert_tag(tag):
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wn.NOUN
    if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wn.VERB
    if tag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    if tag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    return 'trash'


def process_punkt(text):
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        text = text.replace(char, ' ')
    return text


def clean_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stop = stopwords.words("english")
    text = text.strip().lower()
    text = process_punkt(text)
    tokens = re.split("[\s;,]", text)
    tokens = [x for x in tokens if x.isalpha()]
    tokens = [x for x in tokens if len(x) > 3]
    tokens_res = []
    res = nltk.pos_tag(tokens)
    for i in res:
        if convert_tag(i[1]) != 'trash' and i[0] not in stop:
            tokens_res.append(wordnet_lemmatizer.lemmatize(i[0], convert_tag(i[1])))
    return tokens_res


def tokens_to_text(tokens):
    return " ".join(tokens)


def text_to_tokens(text):
    return text.split()


def text_len(text):
    return len(text)


def batch_iterator(data, size, num_epochs, shuffle=True):
