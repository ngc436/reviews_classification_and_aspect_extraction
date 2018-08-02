from gensim.models import Word2Vec
import codecs
import multiprocessing

#model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

IO_DIR = 'data_dir'

class Sentences(object):

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in codecs.open(self.fname, 'r', encoding='utf-8', errors='ignore'):
            yield line.split()

class w2v_model:

    def __init__(self, domain_name, vec_size=200, window=7, min_count=2, verbose=0):
        self.domain = domain_name
        self.vec_size = vec_size
        self.window = window
        self.min_count = min_count
        self.verbose = verbose

    def create_model(self):
        source = '%s/%s/train.txt' % (IO_DIR, self.domain_name)
        target = '%s/%s/w2v_embedding' % (IO_DIR, self.domain_name)
        workers = multiprocessing.cpu_count()
        sg = True
        sentences = Sentences(source, size=self.vec_size, )
        model = Word2Vec()
        print("model is saved: $s")


    def read_data(self):
        raise NotImplementedError
