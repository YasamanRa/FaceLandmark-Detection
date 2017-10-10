from keras.models import Sequential, model_from_config
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
import numpy as np


def model_FC1(train_data, hidden_num_units = 100):

    input_dim = train_data[0].shape[1]
    out_num = train_data[1].shape[1]
    model = Sequential()

    model.add(Dense(hidden_num_units, input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(out_num))

    # optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='mse', optimizer='adam')

    return model

def model_conv3(train_data, num_filters = 32, pool_size=2, hidden_num_units=100):
    #ToDo: try 500 on a GPU

    out_num = train_data[1].shape[1]

    model = Sequential()

    #layer 1
    model.add(Conv2D(kernel_size=3, input_shape=(96, 96, 1), padding="same", filters=num_filters))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    #layer 2
    model.add(Conv2D(kernel_size=2, padding="same", filters=num_filters*2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    #layer 3
    model.add(Conv2D(kernel_size=2, padding="same", filters=num_filters*4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(hidden_num_units))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(out_num))

    model.compile(loss='mse', optimizer='adam')

    return model




