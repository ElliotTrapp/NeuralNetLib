'''
Network.py
Elliot Trapp
18/11/15

The Network class essentially a wrapper class for the model. Holds references to the model and its parameters so they don't have to be passed around.
'''

import numpy as np
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax, Adagrad
import tensorflow as tf
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt 

from Libraries.NeuralNetLib.Utilities import ShowAccuracyPlot, ShowLossPlot, SaveAccuracyPlot, SaveLossPlot, VisualizeModel
from Libraries.DataProcessingLib.TransformData import TrainTestSplit, OneHotEncoder1D
from Libraries.FileManagementLib.Utilities import GetCurrentTimeStamp
from Libraries.FileManagementLib.FileIO import plot_dir
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
from sklearn import metrics
import keras_metrics


import matplotlib.pyplot

class Network:

    def __init__(self,
                 model,
                 train_data,train_labels,
                 valid_data=None,valid_labels=None,
                 test_data=None,test_labels=None,
                 batch_size=None,epochs=1,learning_rate=0.01,decay_rate=None,momentum=None,
                 loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']):
        
        # Set instance variables
        self.model          =   model
        self.train_data     =   train_data
        self.train_labels   =   train_labels
        self.valid_data     =   valid_data
        self.valid_labels   =   valid_labels
        self.test_data      =   test_data
        self.test_labels    =   test_labels
        self.batch_size     =   batch_size
        self.epochs         =   epochs
        self.learning_rate  =   learning_rate
        self.decay_rate     =   decay_rate
        self.momentum       =   momentum
        self.loss           =   loss
        self.metrics        =   metrics
        self.callbacks      =   []

        if self.decay_rate is not None:
            self.decay_rate = decay_rate
        else:
            print("Initializing default decay rate")
            self.decay_rate = self.learning_rate / self.epochs

        self.SetOptimizer(optimizer)
        tensorboard = TensorBoard(log_dir=".logs/{}".format(GetCurrentTimeStamp()))
        self.callbacks.append(tensorboard)

        self.filename = self.GetNetworkFileName()
        self.plot_dir = plot_dir

        self.metrics.extend([keras_metrics.precision(),keras_metrics.recall()])
       
        self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)

    def TrainAndEvaluate(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)

        self.history = self.model.fit(x=self.train_data,y=self.train_labels,batch_size=self.batch_size,
                                    epochs=self.epochs,validation_data=(self.valid_data, self.valid_labels),verbose=1,
                                    shuffle=True,callbacks=self.callbacks)

        metrics = self.model.evaluate(
                    x=self.test_data,
                    y=self.test_labels,
                    batch_size=self.batch_size,
                    verbose=1,steps=None,
                    )

        print("Loss: %.2f" % (metrics[0]))
        print("Validation Accuracy: %.2f" % (metrics[1]))
        print("Validation Precision: %.2f" % (metrics[2]))
        print("Validation Recall: %.2f" % (metrics[3]))

        self.model.save_weights(str(self.filename + '.h5'))  # always save your weights after training or during training
        self.model.save(self.filename)
        self.SavePlots(plot_title=self.filename, plot_dir=self.plot_dir)
        

        sess.close()
        return metrics

    def Train(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)        
        self.history = self.model.fit(x=self.train_data,y=self.train_labels,batch_size=self.batch_size,
                                    epochs=self.epochs,validation_data=(self.valid_data, self.valid_labels),verbose=1,
                                    shuffle=True,callbacks=self.callbacks)

        self.model.save_weights(str(self.filename + '.h5'))  # always save your weights after training or during training
        self.model.save(self.filename)
        self.SavePlots(plot_title=self.filename, plot_dir=self.plot_dir)

        sess.close()
        return self.history

    def Evaluate(self, test_data, test_labels, batch_size=1, steps=None):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)

        score, acc = self.model.evaluate(
                    x=test_data,
                    y=test_labels,
                    batch_size=batch_size,
                    verbose=1,steps=steps,
                    )

        print("Score: %.2f" % (score))
        print("Validation Accuracy: %.2f" % (acc))

        sess.close()
        return score, acc
    
    

    def Predict(self,predict_data,batch_size=None):
        return self.model.predict(x=predict_data,batch_size=batch_size,verbose=1)

    def SetOptimizer(self, optimizer):
        # There is a much better way to do this...
        if optimizer is 'sgd':
            self.optimizer = SGD(lr=self.learning_rate)
        elif optimizer is 'rmsprop':
            self.optimizer = RMSprop(lr=self.learning_rate,decay=self.decay_rate)
        elif optimizer is 'adagrad':
            self.optimizer = Adagrad(lr=self.learning_rate,decay=self.decay_rate)
        elif optimizer is 'adadelta':
            self.optimizer = Adadelta(lr=self.learning_rate,decay=self.decay_rate)
        elif optimizer is 'adamax':
            self.optimizer = Adamax(lr=self.learning_rate,decay=self.decay_rate)
        else:
            print("Initializing default optimizer")
            self.optimizer = Adam(lr=self.learning_rate,decay=self.decay_rate)

    # Prepare data
    def SplitData(self,test_size=0.25,train_size=None,random_state=None,shuffle=True):

        split_train_data, split_valid_data, split_train_labels, split_valid_labels = TrainTestSplit(
        self.train_data,self.train_labels,test_size=test_size,train_size=train_size,
        random_state=random_state,shuffle=shuffle)

        self.train_data     = split_train_data
        self.valid_data     = split_valid_data
        self.train_labels   = split_train_labels
        self.valid_labels   = split_valid_labels

    def OneHotEncodeTrainData(self):
        self.train_data = OneHotEncoder1D(self.train_data)

    def OneHotEncodeTrainLabels(self):
        self.train_labels = OneHotEncoder1D(self.train_labels)

    def OneHotEncodeTestData(self):
        self.test_data = OneHotEncoder1D(self.test_data)

    def OneHotEncodeTestLabels(self):
        self.test_labels = OneHotEncoder1D(self.test_labels)

    # Plotting and visualization
    def ShowPlots(self,plot_title):
        ShowAccuracyPlot(epochs=self.epochs,history=self.history,
                    plot_title=plot_title)
        ShowLossPlot(epochs=self.epochs,history=self.history,
                    plot_title=plot_title)

    def SavePlots(self,plot_dir,plot_title):       
        SaveAccuracyPlot(epochs=self.epochs,history=self.history,
                    plot_title=plot_title,plot_dir=plot_dir)
        SaveLossPlot(epochs=self.epochs,history=self.history,
                plot_title=plot_title,plot_dir=plot_dir)

    def ShowSavePlots(self, plot_title,plot_dir):
        self.SavePlots(plot_title=plot_title,plot_dir=plot_dir)
        self.ShowPlots(plot_title=plot_title)

    def Summarize(self):
        print("Model input shape: ", self.model.input_shape)
        print("Model summary: ", self.model.summary())

    def Visualize(self):
        VisualizeModel(self.model)

    def GetNetworkFileName(self):
        return "{0}_{1}_Epochs={2}_LR={3}_Batch={4}".format(
            str(self.optimizer.__class__.__name__),str(self.loss),self.epochs,self.learning_rate,
                self.batch_size)

    # Setters
    def SetModel(self, model):
        self.model=model

    def SetTrainData(self, train_data):
        self.train_data=train_data

    def SetTrainLabels(self, train_labels):
        self.train_labels=train_labels

    def SetTestData(self, test_data):
        self.test_data=test_data

    def SetTestLabels(self, test_labels):
        self.test_labels=test_labels

    def SetDecayRate(self, decay_rate):
        self.decay_rate = decay_rate

    def SetBatchSize(self, batch_size):
        self.batch_size=batch_size

    def SetNumEpochs(self, num_epochs):
        self.epochs=num_epochs

    def SetLearningRate(self, learning_rate):
        self.learning_rate=learning_rate

    def SetMomentum(self, momentum):
        self.momentum=momentum

    def SetLossFunc(self, loss_func):
        self.loss=loss_func