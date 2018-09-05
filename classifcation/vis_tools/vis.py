# TODO: visualize word embedding weights
import keras

import plotly.graph_objs as go
import matplotlib.pyplot as plt
from keras.utils import plot_model
from plotly.plotly import image
import jupyterlab
import plotly.plotly as py
import os

# getplotlyoffline('http://cdn.plot.ly/plotly-latest.min.js')

# import altair as alt


FIG_DIR = '/'.join([os.getcwd(), 'vis_tools/Figures/'])


# def visualize_weights(data):
#     trace = go.Scatter(
#         x=data[s]
#     )


def plot_distributions(train, test, title='Rating distribution in Amazon dataset'):
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
        title=title,
        barmode='group',
        yaxis=dict(
            title='# of reviews'
        ),
        xaxis=dict(
            title='Rating'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    image.save_as(fig, filename=''.join([FIG_DIR, 'rating_distribution', '.jpeg']))


def plot_len_distribution(column, plot_title):
    plt.hist(column.values.tolist(), 100)
    plt.title(plot_title)
    plt.xlabel('Length of review')
    plt.ylabel('Number of reviews')
    plt.show()

    # alt.Chart(column).mark_bar().encode(
    #     alt.X("Length of review", bin=True),
    #     y='count()',
    # ).serve()

    # trace = [
    #     go.Histogram(
    #         x=column,
    #     )
    # ]
    # layout = go.Layout(
    #     title='Distribution of positive reviews lengths',
    #     xaxis=dict(
    #         title='Length of review'
    #     ),
    #     yaxis=dict(
    #         title='Number of reviews'
    #     )
    # )
    # fig = go.Figure(data=trace, layout=layout)
    # py.plot(fig, filename='simple-histogram')
    # image.save_as(fig, filename=''.join([FIG_DIR, fname, '.jpeg']))


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


class PlotAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        #
        self.acc.append(logs.get('acc'))
        self.i += 1

        plt.plot(self.x, self.acc, label='accuracy')
        plt.legend()
        plt.show()
