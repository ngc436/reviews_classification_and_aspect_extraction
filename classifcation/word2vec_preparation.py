from gensim.models import Word2Vec
import codecs
import multiprocessing
import operator
from classifcation.utils import *

# TODO: implement logging
import logging

import numpy as np

IO_DIR = 'data_dir'


class Sentences(object):

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in codecs.open(self.fname, 'r', encoding='utf-8', errors='ignore'):
            yield line.split()


# maxlen -

class w2v_model:

    def __init__(self):
        self.model = None
        self.embeddings = {}
        self.vector_size = None
        self.matrix = None
        self.vocab = vocab_creation('amazon')

    def create_model(self, domain_name, vec_size=300, window=7, min_count=2):

        source = '%s/%s/train.csv' % (IO_DIR, domain_name)
        target = '%s/%s/w2v_embedding' % (IO_DIR, domain_name)
        workers = multiprocessing.cpu_count()
        sg = True
        sentences = Sentences(source)
        self.model = Word2Vec(sentences, size=vec_size, window=window, min_count=min_count, workers=workers, sg=sg)
        self.model.save(target)
        print("model is saved: %s" % (target))

    # TODO: fix embedding dim
    def read_data(self, domain_name):

        matrix = []
        target = '%s/%s/w2v_embedding' % (IO_DIR, domain_name)
        if not self.model:
            self.model = Word2Vec.load(target)
        for word in self.vocab:
            self.embeddings[word] = list(self.model[word])
            matrix.append(list(self.model[word]))
        self.vector_size = len(self.embeddings)
        self.matrix = np.asarray(matrix)

    def get_word_embedding(self, word):

        try:
            return self.embeddings[word]
        except KeyError:
            return None
