import gensim
import codecs
import multiprocessing

model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

IO_DIR = 'data_dir/'

class Sentences(object):

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in codecs.open(self.fname, 'r', encoding='utf-8', errors='ignore'):
            yield line.split()

class w2v_model:

    def __init__(self, domain_name, vec_size=200, window=7,verbose=0):
        self.domain = domain_name
        self.vec_size = vec_size
        self.window = window
        self.verbose = verbose

    def create_model(self):
        source = IO_DIR + self.domain


    def read_data(self):
        raise NotImplementedError
