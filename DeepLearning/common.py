"""
Common functions for loading datasets etc.

28 May 2016
goker erdogan
https://github.com/gokererdogan
"""

import gzip
import _pickle as pkl
import numpy as np

def load_mnist(path='../datasets'):
    """
    Load MNIST data from disk.
    Data can be downloaded from http://deeplearning.net/data/mnist/mnist.pkl.gz

    Returns
        (2-tuple): training set consisting of training data and class labels
        (2-tuple): validation set consisting of validation data and class labels
        (2-tuple): test set consisting of test data and class labels
    """
    f = gzip.open('{0:s}/mnist.pkl.gz'.format(path), mode='rb')
    train_set, val_set, test_set = np.load(f, encoding='latin1')
    f.close()
    return train_set, val_set, test_set
