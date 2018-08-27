from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

import plotly.offline as offline
import plotly.graph_objs as go
import plotly.plotly as py


def split_data(x, y, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split()


def accuracy():
    raise NotImplementedError


def precision(y_true, y_pred, average='binary'):
    precision_score(y_true, y_pred, average=average)
    raise NotImplementedError


# sensitivity
# average 'binary', 'micro', 'macro', 'weighted', 'samples'
def recall(y_true, y_pred, average='binary'):
    return recall_score(y_true, y_pred, average=average)


def specificity():
    raise NotImplementedError
