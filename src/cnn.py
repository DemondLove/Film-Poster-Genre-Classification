# Import libraries
# import keras
# import numpy as np
# import pandas as pd
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

def developModel():
    '''
    Develop Convolutional Neural Network for MultiClassification

    Returns:
    model (Keras object) - CNN model
    '''
    print('src.cnn.developModel')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(268, 182, 3)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['acc'])

    return model