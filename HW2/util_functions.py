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


def conv2d(input_, weights, biases):
    """ Implements a 2-D convolution operation.
    Will perform using matrix multiplication operations
    Arguments:
    inputs -- an input signal with the dimensions of N X C X H X W
         N is the batch size, C is number of channels, and H and W
         are the input height and width
    weights -- the kernel to perform convolution with. has a dimension of D x h X w
        where D is number of filters to apply, h and w are the kernel dimensions
    biases - bias terms. always 1 dimension array of shape D
    """

    N, C, H, W = input_.shape
    D, _, h, w = weights.shape

    k, i, j = get_im2col_shape(input_.shape, weights.shape, stride=1)
    cols = input_[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(h * w * C, -1)
    print(cols.shape)
    print("done")


def get_im2col_shape(input_shape, kernel_shape, stride=1):
    N, C, H, W = input_shape
    D, _, h, w = kernel_shape

    assert (H - h) % stride == 0
    assert (W - w) % stride == 0

    w_out = (H - h) // stride + 1
    h_out = (W - w) // stride + 1

    i0 = np.repeat(np.arange(h), w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(h_out), w_out)

    j0 = np.tile(np.arange(w), h * C)
    j1 = stride * np.tile(np.arange(w_out), h_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), h * w).reshape(-1, 1)

    return k, i, j


