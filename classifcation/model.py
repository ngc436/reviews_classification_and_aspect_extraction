from keras.layers import Input, Dense, \
    Embedding, Conv2D, MaxPool2D, Reshape, \
    Flatten, Dropout,Concatenate
from keras.optimizers import Adam
from keras.models import Model
import logging
from keras import backend as k

import tensorflow as tf

# memory configuration

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fractoin = 0.5
k.tensorflow_backend.set_session(tf.Session(config=config))
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s % %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class CNN_model:

    def __init__(self, x_train, y_train, ):

        x_train = x_train
        y_train = y_train
        #sequence_length =
        pass


    def train_model(self):
        raise NotImplementedError
