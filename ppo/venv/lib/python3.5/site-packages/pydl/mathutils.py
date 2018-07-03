import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(y):
    return y * (1 - y)


def tanh_prime(y):
    return 1 - np.square(y)