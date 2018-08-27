from numpy import genfromtxt
from classifcation.utils import *
from numpy import genfromtxt
import pandas as pd
from classifcation.preprocess_data import *
from classifcation.vis_tools.vis import *
import plotly

# vocab, train_x, test_x, max_len = read_data('amazon')
# source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'train')
# train_x = codecs.open(source, 'r', 'utf-8')
# source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'test')
# test_x = codecs.open(source, 'r', 'utf-8')
# train_x, test_x = prepare_input_sequences(train_x, test_x, type='w2v_mean')

# y_train = [0, 0, 0, 0, 0]
# for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
#     y_train[int(i) - 1] += 1
#
# y_test = [0, 0, 0, 0, 0]
# for i in genfromtxt('data_dir/amazon/y_test.csv', delimiter=','):
#     y_test[int(i) - 1] += 1
#
# plot_distributions(y_train, y_test)
train_x = []
text = codecs.open('data_dir/amazon/train.csv', 'r', 'utf-8')
for line in text:
    train_x.append(line)
train_y = []
for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
    train_y.append(int(i))
data = {'text': train_x, 'label': train_y}

df = pd.DataFrame(data=data)

df.where(df['label'] == 5, inplace=True)

df['splitted'] = df['text'].apply(str.split)
df['text_len'] = df['splitted'].apply(len)
plot_len_distribution(df['text_len'])
