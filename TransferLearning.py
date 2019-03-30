'''
TransferLearning.py
Elliot Trapp
18/11/15

Utilities for quickly importing and retraining pre-trained networks
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications import mobilenet
from keras.applications.vgg16 import VGG16
import Libraries.DataProcessingLib.AugmentImages as Aug


def GetInceptionV3(num_classes=2, optimizer='adam',loss='categorical_crossentropy'):
    # Set up model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    return base_model

def GetMobilenet(num_classes=2, optimizer='adam',loss='categorical_crossentropy'):
    base_model = mobilenet.MobileNet(include_top=False, weights='imagenet')
    return base_model

def GetVGG16(num_classes=2, optimizer='adam',loss='categorical_crossentropy'):
    base_model = VGG16(include_top=False, weights='imagenet')
    return base_model


def GetPretrainedModel(model,num_classes=2,optimizer='adam',loss='categorical_crossentropy', summary=False):
    ''' Primary interface'''

    if model=='inceptionv3':
        base_model = GetInceptionV3(num_classes=2,optimizer='adam',loss='categorical_crossentropy')
    elif model=='mobilenet':
        base_model = GetMobilenet(num_classes=2,optimizer='adam',loss='categorical_crossentropy')
    elif model=='vgg16':
        base_model = GetVGG16(num_classes=2,optimizer='adam',loss='categorical_crossentropy')
    

    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if summary:
        model.summary()
    
    return model
