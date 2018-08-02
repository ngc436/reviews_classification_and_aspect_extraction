# input file

# create embeddings if don't exist or pass pretrained files
from classifcation.word2vec_preparation import w2v_model
from classifcation.preprocess_data import *
import os
import pandas as pd


def main():
    raw_data_path = '/home/maria/PycharmProjects/Datasets/' \
                    'amazon_review_full_csv/test.csv'
    # raw_data_path = '/home/maria/PycharmProjects/Datasets/' \
    #                 'dataset/review.json'
    # pd.set_option('display.max_colwidth', -1)
    # dataset = pd.read_csv(raw_data_path, header=None)
    # dataset.columns = ['rating', 'subject', 'review']
    # dataset["processed_text"] = dataset["review"].apply(clean_text)
    # dataset["processed_text"] = dataset["processed_text"].apply(tokens_to_text)
    # dataset['rating'].to_csv("data_dir/amazon/y_train.csv")
    # dataset['processed_text'].to_csv("data_dir/amazon/train.csv")

    model = w2v_model()
    model.create_model('amazon')


if __name__ == "__main__":
    main()
