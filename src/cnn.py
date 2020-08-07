# Import libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization

def buildModel():
    '''
    Build Convolutional Neural Network for MultiClassification Architecture

    Returns:
    model (Keras object) - CNN model
    '''
    print('src.cnn.developModel')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(268, 182, 3)))

    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(64, kernel_size=(3, 3), activation='relu'))

    #model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(128, kernel_size=(3, 3), activation='relu'))

    #model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    return model

def calculateCategoricalAccuracy(test_y, predictions):
    '''
    Calculate the pure categorical accuracy of the predictions

    Parameters:
    test_y (np.array) - Vector of actuals
    predictions (np.array) - Vector of predictions produced by the model

    Results:
    categoricalAccuracy (float) - Accuracy Metric
    '''
    print('src.cnn.calculateCategoricalAccuracy')
    
    categoricalAccuracy = tf.keras.metrics.CategoricalAccuracy()

    categoricalAccuracy.update_state(test_y, predictions)

    return categoricalAccuracy.result().numpy()

def calculateTopKCategoricalAccuracy(test_y, predictions, k=2):
    '''
    Calculate the top k categorical accuracy of the predictions
    For example, in k=3, you get credit for having the right answer if the right answer appears in your top 3 probability estimates.

    Parameters:
    test_y (np.array) - Vector of actuals
    predictions (np.array) - Vector of predictions produced by the model
    k (int) - Number Default: 2

    Results:
    topKCategoricalAccuracy (float) - Accuracy Metric
    '''
    print('src.cnn.calculateTopKCategoricalAccuracy')
    
    topKCategoricalAccuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=k)

    topKCategoricalAccuracy.update_state(test_y, predictions)

    return topKCategoricalAccuracy.result().numpy()