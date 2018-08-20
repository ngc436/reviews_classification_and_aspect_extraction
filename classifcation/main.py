# input file

# create embeddings if don't exist or pass pretrained files
from classifcation.word2vec_preparation import w2v_model
from classifcation.utils import *
from sklearn.model_selection import train_test_split
from classifcation.preprocess_data import *
import os
import pandas as pd



def main():

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
    model = w2v_model()
    # model.create_model('amazon')
    print('hello')
    read_data('amazon')

    # raw = ''

    # vocabulary, train, test,


if __name__ == "__main__":
    main()
