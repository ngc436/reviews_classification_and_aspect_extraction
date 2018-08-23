from numpy import genfromtxt
from classifcation.utils import *
from classifcation.preprocess_data import *

vocab, train_x, test_x, max_len = read_data('amazon')
train_x, test_x = prepare_input_sequences(train_x, test_x, type='w2v_mean')


