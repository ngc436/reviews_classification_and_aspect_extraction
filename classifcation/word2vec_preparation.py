from gensim.models import Word2Vec
import codecs
import multiprocessing

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


def vocab_creation(domain_name, maximum_len=0, vocab_size=0):
    try:
        source = '%s/%s/train.csv' % (IO_DIR, domain_name)
    except:
        print("Domain %s doesn't exist" % (domain_name))
    print('Vocabulary initialization')
    total, unique = 0, 0
    word_freqs = {}
    top = 0

    text = codecs.open(source, 'r', 'utf-8')
    for line in text:
        words = line.split()
        if maximum_len > 0 and len(words) > maximum_len:
            continue

        for word in words:
            # flag = bool()
            try:
                word_freqs[word] += 1
            except KeyError:
                unique += 1
                word_freqs[word] = 1
            total += 1


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
            print(self.model)
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
