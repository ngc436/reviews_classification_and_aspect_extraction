# test the ready implemented variant of vae

# https://github.com/DeepmindHub/python-/blob/master/Denoising%20Autoencoder%20VAE.py
# https://github.com/DeepmindHub/python-/blob/master/VAE%20Stacked%20DAE.py
# https://github.com/Toni-Antonova/VAE-Text-Generation/blob/master/vae_nlp.ipynb
# https://github.com/hmishfaq/DDSM-TVAE/blob/master/resnet_new.py
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

# for collaborative filtering
# https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

# loss for vae
# https://blog.keras.io/building-autoencoders-in-keras.html

# VRNN
# https://github.com/jych/nips2015_vrnn
# https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/model.py

# simple example
# https://nicgian.github.io/text-generation-vae/
# https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/

# pics
# https://www.doc.ic.ac.uk/~js4416/163/website/nlp/

# https://github.com/zonetrooper32/VDCNN/blob/keras_version/vdcnn.py

from classifcation.word2vec_preparation import *
from classifcation.utils import *
from sklearn.model_selection import train_test_split
from classifcation.preprocess_data import *
import itertools
import os
from numpy import genfromtxt
from sklearn.model_selection import StratifiedKFold
# from keras.preprocessing import sequence
from keras.utils import to_categorical

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session

set_session(session)

from keras.layers import Input, Dense, \
    Embedding, Conv2D, MaxPool2D, Reshape, \
    Flatten, Dropout, Concatenate, Convolution1D, MaxPooling1D, \
    LSTM, RepeatVector, Activation, Conv1D, GlobalMaxPooling1D, \
    BatchNormalization, Lambda
from keras.engine import Layer, InputSpec
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential
import logging
from keras import losses
from keras import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tqdm import tqdm
import os
import numpy as np
from keras.utils import Sequence, to_categorical
# import skopt
import tensorflow as tf
from tqdm import tqdm
import time
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
from classifcation.vis_tools.vis import *
from keras.callbacks import ModelCheckpoint

IO_DIR = 'data_dir'


def main():
    model = w2v_model()
    model.pretrained_model_from_file('GoogleNews-vectors-negative300.bin')

    # TODO: implement padding
    def return_embeddings(tokens_len=10, set_name='train'):
        data_concat = []
        tokens = vectorize_revs(model, set_name=set_name)
        # example: take only len 20
        data = [x for x in tokens if len(x) == tokens_len]
        print('Concatenation has begun...')
        for x in data:
            data_concat.append(list(itertools.chain.from_iterable(x)))
        print('Concatenation is over')
        data_array = np.array(data_concat)
        np.random.shuffle(data_array)
        return data_array

    train = return_embeddings()
    np.savetxt(IO_DIR+'train_emb.txt', train)
    test = return_embeddings(set_name='test')
    np.savetxt(IO_DIR + 'test_emb.txt',test)
    # TODO: save to file
    res = np.loadtxt(IO_DIR+'test_emb.txt')
    print(test == res)


    batch_size = 500
    original_dim = 3000
    latent_dim = 1000
    intermediate_dim = 1200
    epochs = 200
    epsilon_std = 1.0

    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # placeholder loss
    def zero_loss(y_true, y_pred):
        return K.zeros_like(y_pred)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # we don't use this output, but it has to have the correct shape:
            return K.ones_like(x)

    loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, [loss_layer])
    # added metrics
    vae.compile(optimizer='rmsprop', loss=[zero_loss], metrics=['accuracy'])

    # checkpoint
    cp = [ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)]

    # train
    vae.fit(train, train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test, test),
            callbacks=cp)
    vae.save_weights('vae_mlp_.h5')

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def sent_parse(sentence, mat_shape):
        data_concat = []
        word_vecs = return_embeddings(sentence)
        for x in word_vecs:
            data_concat.append(list(itertools.chain.from_iterable(x)))
        zero_matr = np.zeros(mat_shape)
        zero_matr[0] = np.array(data_concat)
        return zero_matr


if __name__ == '__main__':
    main()
