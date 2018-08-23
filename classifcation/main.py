# input file

# create embeddings if don't exist or pass pretrained files
from classifcation.word2vec_preparation import w2v_model
from classifcation.utils import *
from sklearn.model_selection import train_test_split
from classifcation.preprocess_data import *
import os
import pandas as pd
from keras.preprocessing import sequence
from classifcation.model import CNN_model
from numpy import genfromtxt
from keras.utils import to_categorical


def main():
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

    nn_model = CNN_model()
    # read_data outputs frequency vectors
    vocab, train_x, test_x, max_len = read_data('amazon')

    # TODO: refactor this
    # converting to one-hot representation
    val_list = []
    for i in genfromtxt('data_dir/amazon/y_test.csv', delimiter=','):
        val_list.append([int(i)])
    test_y = np.asarray(val_list).mean(axis=1).astype(int)-1
    test_y = to_categorical(test_y, 5)

    val_list = []
    for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
        val_list.append([int(i)])
    train_y = np.asarray(val_list).mean(axis=1).astype(int) - 1
    train_y = to_categorical(train_y, 5)

    nn_model.create_model(vocab, max_len)
    nn_model.model.get_layer('word_embedding').trainable = False

    source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'train')
    train_x = codecs.open(source, 'r', 'utf-8')
    source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'test')
    test_x = codecs.open(source, 'r', 'utf-8')
    train_x, test_x = prepare_input_sequences(train_x, test_x, type='w2v_mean')
    #train_x, test_x = prepare_input_sequences(train_x, test_x, max_len=max_len, type='freq_seq')

    nn_model.simple_train('amazon', vocab, train_x, train_y,
                          test_x, test_y, max_len)

    # nn_model.train_model(train_x, )
    # transforms a list of num_samples sequences into 2D np.array shape (num_samples, num_timesteps)

    # raw = ''

    # vocabulary, train, test,


if __name__ == "__main__":
    main()
