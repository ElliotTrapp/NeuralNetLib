'''
Models.py
Elliot Trapp
18/11/15

Various Neural Network architectures based on Keras, unless stated otherwise.
'''
from keras.layers import Dense, LSTM, Flatten, BatchNormalization, Dropout, TimeDistributed, MaxPooling2D, Conv2D, Conv3D, Activation
from keras.models import Sequential

def GetBaseCNN(input_shape,num_classes):
   
       model = Sequential()
       model.add(Conv3D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
       model.add(Conv3D(32, (3, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.25))
    
       model.add(Conv3D(64, (3, 3), padding='same', activation='relu'))
       model.add(Conv3D(64, (3, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.25))
    
       model.add(Conv3D(64, (3, 3), padding='same', activation='relu'))
       model.add(Conv3D(64, (3, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(2, 2)))
       model.add(Dropout(0.25))
    
       model.add(Flatten())
       model.add(Dense(512, activation='relu'))
       model.add(Dropout(0.5))
       model.add(Dense(num_classes, activation='softmax'))
       
       return model

def GetBaseLSTM(input_shape, num_classes,num_hidden_nodes):
        model = Sequential()
        model.add(LSTM(num_hidden_nodes, input_shape=input_shape, return_sequences=False))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(num_classes,activation='softmax'))

        return model

def GetRCNN(input_shape,num_classes,num_lstm_nodes=32):
         
        model = Sequential()
        model.add(TimeDistributed(Conv2D(1, (2,2), activation='relu', padding='same', input_shape=input_shape)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(num_lstm_nodes,
          return_sequences=True))
        model.add(Dense(num_classes,activation='softmax'))

        return model

def GetDeepNN(input_shape,num_classes,num_hidden_layers=0,num_hidden_nodes=200):

    model = Sequential()
    model.add(Dense(512, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    for layer in range(num_hidden_layers):
        model.add(Dense(num_hidden_nodes))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation= 'softmax' ))
    
    return model