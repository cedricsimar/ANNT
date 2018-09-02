'''
Contains functions for loading CIFAR-10 data, and preparing data_sets.

Modification of Wolfgang Beyer's code form his tutorial:
"Simple Image Classification Models for the CIFAR-10 dataset using TensorFlow".
'''

import numpy as np
import pickle
import sys

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


def dense_to_one_hot(labels_dense, num_classes):
    '''
    Convert class labels from scalars to one-hot vectors.
    '''
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_CIFAR10_batch(filename):
    '''
    Loads all the data from a single CIFAR-10 batch.

    Args:
        filename: The OS file path towards the CIFAR-10 batch.

    Returns:
        x: An array containing the batch's images.
           (An image is a 3072 array of floats, 3072=32*32*3)
        y: An array containing the batch's labels in one-hot encoding.
           (A label is an 10 array of binary ints, all 0s except one)
    '''
    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            batch_dict = pickle.load(f)
        else:
            batch_dict = pickle.load(f, encoding='latin1')
        x = batch_dict['data']
        y = batch_dict['labels']
        x = x.astype(float)
        y = dense_to_one_hot(np.array(y), 10)
    return x, y


def read_data_sets(validation_size=5000):
    '''
    Reads all the CIFAR-10 data, merging the training batches together.

    Args:
        filename: The OS file path towards the CIFAR-10 batch.

    Returns:
        A dataset.
    '''
    xs = []
    ys = []
    for i in range(1, 6):
        filename = 'CIFAR10_data/data_batch_' + str(i)
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys

    x_test, y_test = load_CIFAR10_batch('CIFAR10_data/test_batch')

    # Split validation set from training set (optionally).
    x_validation = x_train[:validation_size]
    y_validation = y_train[:validation_size]
    x_train = x_train[validation_size:]
    y_train = y_train[validation_size:]

    options = dict(dtype=dtypes.float32, reshape=False, seed=None)

    train = DataSet(x_train, y_train, **options)
    validation = DataSet(x_validation, y_validation, **options)
    test = DataSet(x_test, y_test, **options)

    return base.Datasets(train=train, validation=validation, test=test)
