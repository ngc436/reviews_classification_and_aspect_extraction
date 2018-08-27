# TODO: visualize word embedding weights
import keras

import plotly.graph_objs as go
import matplotlib.pyplot as plt
from keras.utils import plot_model
from plotly.plotly import image

FIG_DIR = 'Figures\\'


# def visualize_weights(data):
#     trace = go.Scatter(
#         x=data[s]
#     )


def plot_distributions(train, test):
    x = ['1', '2', '3', '4', '5']
    trace1 = go.Bar(
        x=x,
        y=train,
        name='Train'
    )
    trace2 = go.Bar(
        x=x,
        y=test,
        name='Test'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group'
    )
    fig = go.Figure(data=data, layout=layout)
    image.save_as(fig, filename=''.join([FIG_DIR, 'rating_distribution', '.jpeg']))


def plot_keras_model(model, fname, show_shapes=False, show_layer_names=True):
    plot_model(model, to_file=fname,
               show_shapes=show_shapes, show_layer_names=show_layer_names)


def plot_training_history(history):
    plt.plot(history.history['acc'])


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        #
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.x, self.losses, label='loss')
        plt.plot(self.x, self.val_losses, label='val_loss')
        plt.legend()
        plt.show()