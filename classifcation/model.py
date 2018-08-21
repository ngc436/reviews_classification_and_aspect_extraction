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
from keras.utils import Sequence, to_categorical
import skopt
import tensorflow as tf
from tqdm import tqdm
import time
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold

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
        self.labels = labels
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.indexes = np.arange(0, self.data.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        part_indixes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.data[part_indixes]
        if self.labels is not None:
            pass


def sentence_batch_generator(data, batch_size=64, num_classes=5, labels=None):
    n_batch = int(np.ceil(len(data) / batch_size))
    batch_count = 0
    # shuffle with labels
    # np.random.shuffle(data)
    indices = np.arange(0, data.shape[0])

    while True:
        if batch_count == n_batch:
            np.random.shuffle(data)
            batch_count = 0



        batch = data[batch_count * batch_size:(batch_count + 1) * batch_size]
        batch_count += 1
        yield batch, y

    # while


def max_margin_loss(y_true, y_pred):
    return k.mean(y_pred)


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

        maxpool_0 = MaxPool2D(pool_size=(max_sentence_len - filter_size[0] + 1, 1), strides=(1, 1), padding='valid')(
            convol_0)
        maxpool_1 = MaxPool2D(pool_size=(max_sentence_len - filter_size[1] + 1, 1), strides=(1, 1), padding='valid')(
            convol_1)
        maxpool_2 = MaxPool2D(pool_size=(max_sentence_len - filter_size[2] + 1, 1), strides=(1, 1), padding='valid')(
            convol_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(utils=2, activation='softmax')(dropout)

        self.model = Model(inputs=inputs, outputs=output, epochs=200)

    def train_model(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=100, max_len=0):
        checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='auto')
        # TODO: tune optimizer parameters
        self.model.compile(optimizer=Adam(lr=1e-4), loss=losses.categorical_crossentropy, metrics=['acc'])
        # transforms a list of num_samples sequences into 2D np.array shape (num_samples, num_timesteps)
        train_x = sequence.pad_sequences(x_train, maxlen=max_len)
        print('Size of training set: %i' % len(train_x))
        test_x = sequence.pad_sequence(x_test, maxlen=max_len)
        print('Size of test set: %i' % len(test_x))
        sen_gen = sentence_batch_generator(x_train, batch_size)
        batches_num_epoch = 1000
        for i in range(epochs):
            start = time.time()
            loss, max_margin_loss = 0., 0.
            for b in tqdm(range(batches_num_epoch)):
                sen_input = sen_gen.next()
                batch_loss, batch_max_margin_loss = self.model.train_on_batch(sen_input, np.ones((batch_size, 1)))
                loss += batch_loss / batches_num_epoch
                max_margin_loss += batch_max_margin_loss / batches_num_epoch
        total_time = time.time() - start

        print('Model is being trained...')
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[checkpoint], validation_data=(x_test, y_test))

    def predict(self):
        raise NotImplementedError


class LSTM_model(Base_Model):

    def __init__(self):
        # sequence_length =
        model = None
