from gensim.models import Word2Vec, KeyedVectors
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


# TODO: implement me
def load_google_w2v(fname, vocab, text):
    embeddings = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, l_size = map(int, header.split())
        bin_len = np.dtype('float32').itemsize * l_size
        #
        # for sentence in text:
        #     if word in sentence:
        #         if word in


# use in case of words absence
# returns model if the dict is not specified
def ready_model_train(model, sentences_with_unknown_words, dict_of_unknown=None):
    model.train(sentences_with_unknown_words)
    if not dict_of_unknown:
        embeddings = {}
        for word in dict_of_unknown:
            embeddings[word] = model.wv[word]
        return embeddings
    return model


# numpy array of arrays (texts as images)
def prepare_emb_input(w2v_dict, text, max_len, emb_dim=300):
    list_of_np_embeddings = []
    input_shape = (max_len, emb_dim)
    for sentence in text:
        tmp_arr = np.zeros(input_shape)
        for ind, word in enumerate(sentence.split()):
            tmp_arr[ind] = w2v_dict[word]
        list_of_np_embeddings.append(tmp_arr)
    return np.array(list_of_np_embeddings)

# def get_sentences_with_unknown_words(model, ):


# maxlen -

def get_embeddings(vocab, w2v_model):
    emb_dict = {}
    undefined = []
    for word in vocab:
        if word in w2v_model.model.wv:
            emb_dict[word] = w2v_model.model.wv[word]
        else:
            undefined.append(word)
    return emb_dict, undefined

def vectorize_revs(revs, w2v_model):
    vectorized = []
    for rev in revs:
        words = rev.split()
        vect = []
        for word in words:
            vect.append()


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

    def model_from_file(self, domain_name):
        self.model = KeyedVectors.load_word2vec_format('%s/%s/w2v_embedding' % (IO_DIR, domain_name))

    def pretrained_model_from_file(self,fname):
        self.model = KeyedVectors.load_word2vec_format('%s/%s' % (IO_DIR, fname), binary=True)

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

    def create_embedding_matrix(self):
        raise NotImplementedError

    def get_w2v_mean(self, text, size=300):
        vec = np.zeros(size).reshape((1, size))
        num_words_in_vocab = 0
        for word in text.split():
            try:
                vec += self.model[word]
                num_words_in_vocab += 1
            except KeyError:
                continue
        if num_words_in_vocab != 0:
            vec /= num_words_in_vocab
        return vec
