# input file

# create embeddings if don't exist or pass pretrained files
from classifcation.word2vec_preparation import w2v_model
from classifcation.utils import *
from sklearn.model_selection import train_test_split
from classifcation.preprocess_data import *
import os
import pandas as pd
from keras.preprocessing import sequence
from classifcation.model import CNN_model, LSTM_model, VDCNN
from numpy import genfromtxt
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.datasets import imdb


def main():
    NUM_CLASSES = 5
    SEQUENCE_MAX_LEN = 1024

    # TODO: load all data + implement cross-validation

    # TODO: look at distributions of ratings in train/test sets
    # raw_data_path = '/home/maria/PycharmProjects/Datasets/' \
    #                 'amazon_review_full_csv/test.csv'
    # dataset1 = pd.read_csv(raw_data_path, header=None)
    # dataset1.columns = ['rating', 'subject', 'review']
    # raw_data_path = '/home/maria/PycharmProjects/Datasets/' \
    #                 'amazon_review_full_csv/train.csv'
    # dataset2 = pd.read_csv(raw_data_path, header=None)
    # dataset2.columns = ['rating', 'subject', 'review']
    # data = [dataset1, dataset2]
    # dataset = pd.concat(data)
    #
    # dataset["processed_text"] = dataset["review"].apply(clean_text)
    # print('Text is clean!')
    # dataset["processed_text"] = dataset["processed_text"].apply(tokens_to_text)
    # train, test = train_test_split(dataset, test_size=0.2)
    # test['processed_text'].to_csv("data_dir/amazon/test.csv", index=False)
    # test['rating'].to_csv("data_dir/amazon/y_test.csv", index=False)
    # print('test sets are ready!')
    # train['processed_text'].to_csv("data_dir/amazon/train.csv", index=False)
    # train['rating'].to_csv("data_dir/amazon/y_train.csv", index=False)
    # print('train sets is ready!')

    # pd.set_option('display.max_colwidth', -1)
    # dataset = pd.read_csv(raw_data_path, header=None)
    # dataset.columns = ['rating', 'subject', 'review']
    # dataset["processed_text"] = dataset["review"].apply(clean_text)
    # dataset["processed_text"] = dataset["processed_text"].apply(tokens_to_text)
    # dataset['rating'].to_csv("data_dir/amazon/y_train.csv")
    # dataset['processed_text'].to_csv("data_dir/amazon/train.csv")
    #
    # model = w2v_model()
    # model.create_model('amazon')

    # nn_model = CNN_model()
    # # read_data outputs frequency vectors
    vocab, train_x, test_x, max_len = read_data('amazon')
    #
    # # TODO: refactor this
    # converting to one-hot representation
    val_list = []
    for i in genfromtxt('data_dir/amazon/y_test.csv', delimiter=','):
        val_list.append([int(i)])
    test_y = np.asarray(val_list).mean(axis=1).astype(int) - 1
    test_y = to_categorical(test_y, 5)

    val_list = []
    for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
        val_list.append([int(i)])
    train_y = np.asarray(val_list).mean(axis=1).astype(int) - 1
    train_y = to_categorical(train_y, 5)

    ## for vdcnn


    # train_x = codecs.open('%s/%s/train.csv' % (IO_DIR, 'amazon'), mode='r', encoding='utf-8')
    # train_x = get_sequence(train_x)
    # train_x = sequence.pad_sequences(train_x, maxlen=SEQUENCE_MAX_LEN, padding='post', truncating='post')
    #
    # test_x = codecs.open('%s/%s/test.csv' % (IO_DIR, 'amazon'), mode='r', encoding='utf-8')
    # test_x = get_sequence(test_x)
    # test_x = sequence.pad_sequences(test_x, maxlen=SEQUENCE_MAX_LEN, padding='post', truncating='post')

    # nn_model.create_model(vocab, max_len)
    # nn_model.model.get_layer('word_embedding').trainable = False
    #
    # print('Transforming train data to list')
    # source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'train')
    # train_x = codecs.open(source, 'r', 'utf-8')
    # new_train_x = []
    # for line in train_x:
    #     new_train_x.append(line.split())
    # train_x = new_train_x
    #
    # print('Transforming test data to list')
    # source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'test')
    # test_x = codecs.open(source, 'r', 'utf-8')
    # new_test_x = []
    # for line in test_x:
    #     new_test_x.append(line.split())
    # test_x = new_test_x
    #
    # # train_x, test_x = prepare_input_sequences(train_x, test_x, type='w2v_mean')
    #
    # train_x, test_x = prepare_input_sequences(train_x, test_x, max_len=max_len, type='freq_seq', max_num_of_words=100000)
    #
    # np.save('%s/%s/%s.npy' % (IO_DIR, 'amazon', 'train_x_pad_100000'), train_x)
    # np.save('%s/%s/%s.npy' % (IO_DIR, 'amazon', 'test_x_pad_100000'), test_x)

    train_x = np.load('%s/%s/%s.npy' % (IO_DIR, 'amazon', 'train_x_pad'))
    test_x = np.load('%s/%s/%s.npy' % (IO_DIR, 'amazon', 'test_x_pad'))

    # cross validation section
    # skf = StratifiedKFold(indices, n_folds=n_folds, shuffle=True)

    # nn_model.simple_train('amazon', vocab, train_x, train_y, test_x,
    #                       test_y, max_len)

    # nn_model.train_model(train_x, )
    # transforms a list of num_samples sequences into 2D np.array shape (num_samples, num_timesteps)

    raw = ''

    # vocabulary, train, test,

    # create LSTM_CNN model

    # nn_model = LSTM_model()
    # nn_model.create_model_with_conv_layer(len(vocab), max_len)
    # nn_model.train_model(vocab, train_x, train_y, test_x, test_y, max_len)

    ##### test imdb
    # max_features = 10000
    # maxlen = 500
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)
    # nn_model = CNN_model()
    # nn_model.create_imdb_model()
    # nn_model.fit_imdb_model(x_train, y_train, x_test, y_test)


    train_x = sequence.pad_sequences(train_x, maxlen=SEQUENCE_MAX_LEN, padding='post', truncating='post')

    test_x = sequence.pad_sequences(test_x, maxlen=SEQUENCE_MAX_LEN, padding='post', truncating='post')

    nn_model = VDCNN()
    nn_model.create_model()
    nn_model.train_model(train_x, train_y, test_x, test_y, vocab)


if __name__ == "__main__":
    main()
