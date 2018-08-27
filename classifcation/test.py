from numpy import genfromtxt
from classifcation.utils import *
from classifcation.preprocess_data import *
from classifcation.vis_tools.vis import *
import plotly

# vocab, train_x, test_x, max_len = read_data('amazon')
# source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'train')
# train_x = codecs.open(source, 'r', 'utf-8')
# source = '%s/%s/%s.csv' % (IO_DIR, 'amazon', 'test')
# test_x = codecs.open(source, 'r', 'utf-8')
# train_x, test_x = prepare_input_sequences(train_x, test_x, type='w2v_mean')

plotly.tools.set_credentials_file(username='MKhod', api_key='wHS62oM2tyOtpJFhwJ1Y')

y_train = [0, 0, 0, 0, 0]
for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
    y_train[int(i) - 1] += 1

y_test = [0, 0, 0, 0, 0]
for i in genfromtxt('data_dir/amazon/y_train.csv', delimiter=','):
    y_test[int(i) - 1] += 1

plot_distributions(y_train, y_test)

