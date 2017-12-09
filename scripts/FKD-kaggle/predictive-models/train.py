import argparse
import sys
import data
import model
from model import model_FC1
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import datetime
import numpy as np
from keras.layers.pooling import MaxPooling1D, AveragePooling1D


def train_model(model_func, save_path, data_params, epochs=400, save_model=False):

    train, X_test = data.data(**data_params)
    model = model_func(train)
    model.summary()

    X_train, y_train = train[0], train[1]

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    checkpointer = ModelCheckpoint(filepath=os.path.join(save_path, "weights.hdf5"), verbose=1, save_best_only=True)
    # earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

    print('model training')
    hist = model.fit(X_train, y_train, batch_size=128, epochs=epochs, verbose=2, validation_split=0.2,
              callbacks=[checkpointer])

    model.load_weights(os.path.join(save_path, "weights.hdf5"))
    
    np.savetxt(os.path.join(save_path,'model_top_vgg_flip_loss.csv'), hist.history['loss'])
    np.savetxt(os.path.join(save_path,'model_top_vgg_flip_val_loss.csv'), hist.history['val_loss'])
    if save_model:
        model.save(os.path.join(save_path, "model.h5"))


if __name__ == "__main__":

    # ToDo: set these variables before running this script
    #model_func, exp_name = model.model_FC1, "single_hidden_layer"
    #model_func, exp_name = model.model_conv3, "three_layer_convolution"
    data_params = {"reshape":False, "g2rgb":True}
    model_func, exp_name = model.model_vgg, "VGG-2fc"
    #model_func, exp_name = model.model_resnet50, "ResNet50"

    train_model(model_func, os.path.join("results/", exp_name), data_params, epochs=1000, save_model=True)







