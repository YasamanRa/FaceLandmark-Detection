import os
import numpy as np
import pandas as pd

dtdir = '../../../datasets/FKD-kaggle'
img_width = 96
img_height = 96

def gray_to_rgb(X):
    X = X.reshape(-1, 96, 96, 1)
    
    ret = np.empty((X.shape[0], img_width, img_height, 3), dtype=np.float32)
    ret[:, :, :, 0] = X[:, :, :, 0]
    ret[:, :, :, 1] = X[:, :, :, 0]
    ret[:, :, :, 2] = X[:, :, :, 0]
    return ret


def data(drop=True, cols=None, reshape=False, g2rgb=False):

    test_set = pd.read_csv(os.path.join(dtdir, 'test.csv'))
    train_set = pd.read_csv(os.path.join(dtdir, 'training.csv'))
    train_set['Image'] = train_set['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    test_set['Image'] = test_set['Image'].apply(lambda im: np.fromstring(im, sep=' '))


    if cols:  # get a subset of columns
        train_set = train_set[list(cols) + ['Image']]
        test_set = test_set[list(cols) + ['Image']]

    print(train_set.count())

    if drop:
        train_set = train_set.dropna()
        test_set = test_set.dropna()

    X_train = np.vstack(train_set['Image'].values) / 255.
    X_train = X_train.astype(np.float32)
    X_test = np.vstack(test_set['Image'].values) / 255.
    X_test = X_test.astype(np.float32)

    y_train = train_set[train_set.columns[:-1]].values
    y_train = (y_train - 48) / 48.  # scale target coordinates to [-1, 1]
    y_train = y_train.astype(np.float32)
    
    if reshape:
        X_train = X_train.reshape(-1, 96, 96, 1)
        X_test = X_test.reshape(-1, 96, 96, 1)
        
    if g2rgb:
        X_train = gray_to_rgb(X_train)
        X_test = gray_to_rgb(X_test)

    print('Train Shape:', X_train.shape, y_train.shape)
    print('Test Shape:', X_test.shape)

    return (X_train, y_train), X_test


if __name__ == "__main__":
    data()






