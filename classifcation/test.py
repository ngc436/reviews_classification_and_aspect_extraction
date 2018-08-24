from numpy import genfromtxt
from classifcation.utils import *
from classifcation.preprocess_data import *

# vocab, train_x, test_x, max_len = read_data('amazon')
source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'train')
train_x = codecs.open(source, 'r', 'utf-8')
source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'test')
test_x = codecs.open(source, 'r', 'utf-8')
train_x, test_x = prepare_input_sequences(train_x, test_x, type='w2v_mean')

y_train = [0,0,0,0,0]
for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
    y_train.append([int(i)])

y_test = [0,0,0,0,0]
for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
    y_train.append([int(i)])


