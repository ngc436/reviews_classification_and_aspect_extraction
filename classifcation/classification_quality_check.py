from sklearn.model_selection import train_test_split

import plotly.offline as offline
import plotly.graph_objs as go
import plotly.plotly as py

def split_data(x,y,test_size=0.2,random_state=42):
    x_train, x_test, y_train, y_test = train_test_split()

def accuracy():
    raise NotImplementedError

def precision():
    raise NotImplementedError

# sensitivity
def recall():
    raise NotImplementedError

def specificity():
    raise NotImplementedError