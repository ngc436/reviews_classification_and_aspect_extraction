import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from collections import namedtuple
import multiprocessing

IO_DIR = 'data_dir'

class d2v_model:

    def __init__(self):
        self.model = None


    def create_model(self):
        assert gensim.models.Doc2Vec.FAST_VERSION > -1


