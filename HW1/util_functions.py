import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return relu(x) / np.abs(x)


def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)
