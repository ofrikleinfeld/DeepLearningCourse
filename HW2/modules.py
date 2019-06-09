import numpy as np
import HW2.util_functions as util_functions


class NetworkModule(object):

    def __call__(self, input_):
        raise NotImplementedError("Sub class must implement forward pass")

    def backward(self, *args):
        raise NotImplementedError("Sub class must implement backward pass")


class NetworkModuleWithParams(NetworkModule):

    def __init__(self):
        self.weights = None
        self.biases = None

    def init_weights(self, *args):
        raise NotImplementedError("Sub class must implement an initialization method for weights")

    def set_weights(self, new_weights):
        self.weights = new_weights

    def set_biases(self, new_biases):
        self.biases = new_biases


class Conv2d(NetworkModuleWithParams):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2d, self).__init__()
        self.init_weights(in_channels, out_channels, kernel_size)
        self.stride = stride

    def init_weights(self, in_channels, out_channels, kernel_size):
        # naive initialization from uniform distribution
        self.weights = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
        self.biases = np.random.rand(out_channels)

    def __call__(self, input_):
        return util_functions.conv2d(input_, self.weights, self.biases, self.stride)


class Linear(NetworkModuleWithParams):

    def __init__(self, in_dimension, out_dimension):
        super(Linear, self).__init__()
        self.init_weights(in_dimension, out_dimension)

    def init_weights(self, in_dimension, out_dimension):
        # naive initialization from uniform distribution
        self.weights = np.random.rand(out_dimension, in_dimension)
        self.biases = np.random.rand(out_dimension)

    def __call__(self, input_):
        return util_functions.linear(input_, self.weights, self.biases)


class Relu(NetworkModule):

    def __call__(self, input_):
        return util_functions.relu(input_)

    def backward(self, input_):
        return util_functions.relu_derivative(input_)


class Softmax(NetworkModule):

    def __call__(self, input_):
        return util_functions.softmax(input_)

    def backward(self, input_, label):
        return self(input_) - label


class Sigmoid(NetworkModule):

    def __call__(self, input_):
        return util_functions.sigmoid(input_)

    def backward(self, input_):
        return util_functions.sigmoid_derivative(input_)
