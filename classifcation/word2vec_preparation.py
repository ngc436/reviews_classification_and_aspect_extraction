import gensim
import codecs

model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

IO_DIR = 'data_dir/'

class Sentences(object):

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in codecs.open(self.fname, 'r', 'utf-8'):
            yield line.split()

class w2v_model:

    def __init__(self, domain_name):
        domain = domain_name

    def train(self):
        source = ""
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError
