from keras.layers import Input, Dense, \
    Embedding, Conv2D, MaxPool2D, Reshape, \
    Flatten, Dropout, Concatenate
from keras.optimizers import Adam
from keras.models import Model
import logging
from keras import losses
from keras import backend as k
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import os
import numpy as np
from keras.utils import Sequence

import tensorflow as tf

# memory configuration

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
k.tensorflow_backend.set_session(tf.Session(config=config))
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s % %(levelname)s %(message)s')
# TODO: add logging info messages
logger = logging.getLogger(__name__)


# data is passed as np array
class BatchGenerator(Sequence):
    def __init__(self, data, batch_size=64, labels=None, num_classes=5):
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.indexes = np.arange(0, self.data.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        part_indixes = self.indexes


def sentence_batch_generator(data, batch_size=64, num_classes=5):
    n_batch = len(data) / batch_size
    batch_count = 0
    np.random.shuffle(data)


class Base_Model:

    def create_model(self, *args):
        raise NotImplementedError

    def train_model(self, *args):
        raise NotImplementedError

    def predict(self, *args):
        raise NotImplementedError


# TODO: perform model optimization

class CNN_model(Base_Model):

    def __init__(self):
        # sequence_length =
        model = None

    # max_sentence_len = 0 means no limit on num of words during training
    def create_model(self, vocabulary, max_sentence_len=0, embedding_dim=None,
                     num_conv_filters=512, filter_size=None, drop=0.5):
        if filter_size is None:
            filter_size = [3, 4, 5]
        vocab_size = len(vocabulary)
        inputs = Input(shape=(max_sentence_len,), dtype='int32', name='reviews_input')
        # TODO: find out what is neg_input
        word_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                   mask_zero=True, name='word_embedding')
        # TODO: check input type and the need of reshaping
        # TODO: thy other activation function
        convol_0 = Conv2D(num_conv_filters, kernel_size=(filter_size[0], embedding_dim),
                          padding='valid', kernel_initializer='normal', activation='relu'
                          )(word_embedding)
        convol_1 = Conv2D(num_conv_filters, kernel_size=(filter_size[1], embedding_dim),
                          padding='valid', kernel_initializer='normal', activation='relu'
                          )(word_embedding)
        convol_2 = Conv2D(num_conv_filters, kernel_size=(filter_size[2], embedding_dim),
                          padding='valid', kernel_initializer='normal', activation='relu'
                          )(word_embedding)

        maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_size[0] + 1, 1), strides=(1, 1), padding='valid')(
            convol_0)
        maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_size[1] + 1, 1), strides=(1, 1), padding='valid')(
            convol_1)
        maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_size[2] + 1, 1), strides=(1, 1), padding='valid')(
            convol_2)

        # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten

        self.model = Model(inputs=inputs, outputs=outputs, epochs=200)

    def train_model(self, x_train, y_train, epochs=100, batch_size=100):
        checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='auto')
        # TODO: tune optimizer params
        self.model.compile(optimizer=Adam(lr=1e-4), loss=losses.categorical_crossentropy, metrics=['acc'])
        print('Model is being trained...')
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[checkpoint], validation_data=(X_test))

    def predict(self):
        raise NotImplementedError


class LSTM_model(Base_Model):
    pass