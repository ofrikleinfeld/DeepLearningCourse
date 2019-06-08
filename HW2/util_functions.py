import numpy as np


def softmax(x):

    if len(x.shape) == 1:
        z = x - np.max(x)
        exp_z = np.exp(z)
        exp_sum = np.sum(exp_z)

    else:
        z = x - np.max(x, axis=1)[:, np.newaxis]
        exp_z = np.exp(z)
        exp_sum = np.sum(exp_z, axis=1)[:, np.newaxis]

    return exp_z / exp_sum


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return relu(x) / np.abs(x + 1e-5)


def sigmoid(x):
    z = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return z


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))