from keras.layers import Input, Dense, \
    Embedding, Conv2D, MaxPool2D, Reshape, \
    Flatten, Dropout, Concatenate
from keras.optimizers import Adam
from keras.models import Model
import logging
from keras import backend as k

import tensorflow as tf

# memory configuration

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# k.tensorflow_backend.set_session(tf.Session(config=config))
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s % %(levelname)s %(message)s')
# TODO: add logging info messages
logger = logging.getLogger(__name__)


class Base_Model:

    def create_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class CNN_model(Base_Model):

    def __init__(self):
        # sequence_length =
        model = None

    # TODO: change to more clever implementation
    def create_model(self, data_len, num_conv_filters=512, filter_size=[3, 4, 5],
                     ):
        inputs = Input(shape=(data_len,), dtype='int32')
        embedding = Embedding(input_dim=vocabulary_size,
                              output_dim=embedding_dim)
        # TODO: check input type
        convol_0 = Conv2D(num_conv_filters, kernel_size=filter_size[0],
                          )
        # TODO: implement model

        self.model = Model(inputs=inputs, outputs=outputs, epochs=200)

    def train_model(self, x_train, y_train, batch_size=100):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self):
        raise NotImplementedError
