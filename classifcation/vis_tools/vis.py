# TODO: visualize word embedding weights

import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from keras.utils import plot_model


def visualize_weights():
    trace = go.Scatter(
        x=data[s]
    )

def plot_keras_model(model, fname, show_shapes=False, show_layer_names=True):
    plot_model(model, to_file=fname,
               show_shapes=show_shapes, show_layer_names=show_layer_names)
