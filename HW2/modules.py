import numpy as np
import HW2.util_functions as util_functions


class NetworkModule(object):

    def __init__(self):
        self.z = None
        self.a = None
        self.layer_grad = None

    def __call__(self, input_):
        raise NotImplementedError("Sub class must implement forward pass")

    def backward(self, *args):
        raise NotImplementedError("Sub class must implement backward pass")


class NetworkModuleWithParams(NetworkModule):

    def __init__(self):
        super(NetworkModuleWithParams, self).__init__()
        self.weights = None
        self.biases = None
        self.w_grad = None
        self.b_grad = None

    def init_weights(self, *args):
        raise NotImplementedError("Sub class must implement an initialization method for weights")

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_layer_grad(self):
        return self.layer_grad

    def get_w_grad(self):
        return self.w_grad

    def get_b_grad(self):
        return self.b_grad

    def set_weights(self, new_weights):
        self.weights = new_weights

    def set_biases(self, new_biases):
        self.biases = new_biases


# class Conv2d(NetworkModuleWithParams):
#
#     def __init__(self, in_channels, out_channels, kernel_size, activation_layer, stride=1, padding=0):
#         super(Conv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.init_weights(in_channels, out_channels, kernel_size)
#         self.activation_layer = activation_layer
#         self.stride = stride
#         self.padding = padding
#         self.input_as_col = None
#         self.a_minus_1 = None
#         self.z = None
#         self.a = None
#
#     def init_weights(self, in_channels, out_channels, kernel_size):
#         # naive initialization from uniform distribution
#         self.weights = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)
#         self.biases = np.random.rand(out_channels)
#
#     def __call__(self, input_):
#         self.a_minus_1 = input_
#         self.z, self.input_as_col_ = util_functions.conv2d(input_, self.weights, self.biases, self.stride)
#         self.a = self.activation_layer(self.z)
#         return self.a
#
#     def backward(self, next_layer_grad):
#         D, C, h, w = self.weights.shape
#
#         self.b_grad = np.sum(next_layer_grad, axis=(0, 2, 3))
#         self.b_grad = self.b_grad.reshape(D, -1)
#
#         next_layer_grad_reshaped = next_layer_grad.transpose(1, 2, 3, 0).reshape(D, -1)
#         self.w_grad = next_layer_grad_reshaped @ self.input_as_col.T
#         self.w_grad = self.w_grad.reshape(self.weights.shape)
#
#         weights_reshape = self.weights.reshape(D, -1)
#         layer_grad_col = weights_reshape.T @ next_layer_grad_reshaped
#         self.layer_grad = util_functions.col2im_indices(layer_grad_col, self.a_minus_1.shape, h, w, padding=self.padding, stride=self.stride)
#
#         return self.layer_grad


class Linear(NetworkModuleWithParams):

    def __init__(self, in_dimension, out_dimension):
        super(Linear, self).__init__()
        self.init_weights(in_dimension, out_dimension)
        self.a_minus_1 = None

    def init_weights(self, in_dimension, out_dimension):
        # naive initialization from uniform distribution
        self.weights = np.random.rand(out_dimension, in_dimension)
        self.biases = np.random.rand(out_dimension)

    def __call__(self, input_):
        self.a_minus_1 = input_
        self.z = util_functions.linear(input_, self.weights, self.biases)
        return self.z

    def backward(self, next_layer_weights, next_layer_grad):
        self.layer_grad = next_layer_grad @ next_layer_weights
        self.w_grad = self.layer_grad.T @ self.a_minus_1
        self.b_grad = self.layer_grad
        return self.layer_grad


class Relu(NetworkModule):

    def __call__(self, z):
        self.z = z
        self.a = util_functions.relu(z)
        return self.a

    def backward(self):
        self.layer_grad = util_functions.relu_derivative(self.z)
        return self.layer_grad


class Softmax(NetworkModule):

    def __call__(self, z):
        self.z = z
        self.a = util_functions.softmax(z)
        return self.a

    def backward(self, label):
        # assuming cross entropy loss
        self.layer_grad = self.a - label
        return self.layer_grad


class Sigmoid(NetworkModule):

    def __call__(self, z):
        self.z = z
        self.a = util_functions.sigmoid(z)
        return self.a

    def backward(self):
        self.layer_grad = util_functions.sigmoid_derivative(self.z)
        return self.layer_grad


class Flatten(NetworkModule):

    def __call__(self, z):
        self.z = z
        batch_size = z.shape[0]
        self.a = np.reshape(z, (batch_size, -1))
        return self.a

    def backward(self, next_layer_weights, next_layer_grad):
        layer_grad_flatten = next_layer_grad @ next_layer_weights
        N, C, H, W = self.z.shape
        self.layer_grad = np.reshape(layer_grad_flatten, (N, C, H, W))
        return self.layer_grad
