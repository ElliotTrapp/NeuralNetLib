'''
Utilities.py
Elliot Trapp
18/11/15

Set of utilities for producing and managing batches of NN training data
'''

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import tensorflow as tf
import time, datetime
from Libraries.DataProcessingLib.VisualizeData import SavePlot
from abc import ABC, abstractclassmethod, abstractmethod
from ann_visualizer.visualize import ann_viz

# Constants
default_plot_dir=r'../../Plots/'
default_plot_title=""

# Source: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
def next_batch(data, labels, batch_size):
    '''
    Return a total of `batch_size` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Source: http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
def next_series_batch(data, batch_size, num_steps):

    data = tf.convert_to_tensor(data, name="data", dtype=tf.int32)

    data_len = tf.size(data)
    batch_len = data_len // batch_size
    data = tf.reshape(data[0: batch_size * batch_len],
                    [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    
    return x, y

def VisualizeModel(model):
    ann_viz(model, filename='network.gv', title="ModelArchitecture")

def PlotAccuracy(epochs, history, plot_title=default_plot_title):
    fig = plt.figure()
    figure(figsize=(7,7))
    plt.tight_layout()
    plt.title(plot_title)
    plt.plot(range(1,epochs+1),history.history['val_acc'],label='validation',lw=2)
    plt.plot(range(1,epochs+1),history.history['acc'],label='training',lw=2)
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1,epochs])
    plt.grid(True)
    return fig


def SaveAccuracyPlot(epochs, history, plot_title=default_plot_title, plot_dir=default_plot_dir):
    plot_title = "Accuracy_" + plot_title
    fig = PlotAccuracy(epochs=epochs,history=history,plot_title=plot_title)
    SavePlot(plot_title=plot_title, plot_dir=plot_dir)
    plt.close()
    plt.close(fig)

def ShowAccuracyPlot(epochs, history, plot_title=default_plot_title):
    plot_title = "Accuracy_" + plot_title
    fig = PlotAccuracy(epochs,history,plot_title=plot_title)
    plt.show()
    plt.close()
    plt.close(fig)

def PlotLoss(epochs, history, plot_title=default_plot_title):
    fig = plt.figure()
    figure(figsize=(7,7))
    plt.plot(range(1,epochs+1), history.history['loss'],label='loss',lw=2)
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim([1,epochs])
    plt.title(plot_title)
    plt.grid(True)
    return fig

def SaveLossPlot(epochs, history, plot_title=default_plot_title, plot_dir=default_plot_dir):
    plot_title = "Loss_" + plot_title
    fig = PlotLoss(epochs=epochs,history=history,plot_title=plot_title)
    SavePlot(plot_title=plot_title, plot_dir=plot_dir)
    plt.close()
    plt.close(fig)

def ShowLossPlot(epochs, history, plot_title=default_plot_title,):
    plot_title = "Loss_" + plot_title
    fig = PlotLoss(epochs=epochs,history=history,plot_title=plot_title)
    plt.show()
    plt.close()
    plt.close(fig)

def CompareModels(epochs, labeled_histories, plot_title=default_plot_title, plot_dir=None):
    
    figure(figsize=(7,7))
    plot_title = "CompareModels_" + plot_title
    for label, history in labeled_histories:
        plt.plot(range(1,epochs+1),history.history['val_acc'],label=label)
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.xlim([0,epochs])
    plt.ylabel('accuracy on validation set')
    plt.grid(True)
    plt.title(plot_title)
    
    if plot_dir is not None:
        save_plot(plot_title=default_plot_title, plot_dir=plot_dir)
    
    plt.show()
    plt.close(fig)