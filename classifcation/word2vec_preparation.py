from gensim.models import Word2Vec
import codecs
import multiprocessing
import operator

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
    print('Vocabulary initialization...')
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
    print('Total amount of words %i with %i unique ones' % (total, unique))
    sorted_freq = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    # TODO: simplify this part
    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_freq:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('Vocabulary size is %i' % vocab_size)

    ofile = codecs.open('%s/%s/vocab' % (IO_DIR, domain_name), mode='w', encoding='utf-8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            ofile.write(word + '/t' + str(0) + '\n')
            continue
        ofile.write(word +'/t' + str(word_freqs[word]) + '\n')
    ofile.close()
    print('Vocabulary is successfully created')

    return vocab


def read_set(domain_name, set_type, vocab, max_len):
    assert set_type in {'train', 'test'}
    source = '%s/%s/%s.csv' % (IO_DIR, domain_name, set_type)

    max_len_x = 0


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
