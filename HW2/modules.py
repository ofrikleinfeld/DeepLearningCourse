import numpy as np
import HW2.util_functions as util_functions


class NetworkModule(object):

    def __init__(self):
        self.layer_input = None
        self.layer_output = None
        self.layer_grad = None

    def __call__(self, input_):
        raise NotImplementedError("Sub class must implement forward pass")

    def backward(self, next_layer_grad):
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

    def xavier_initialization(self, in_dimension, out_dimension):
        self.weights = np.random.normal(loc=0, scale=in_dimension, size=(out_dimension, in_dimension))
        self.biases = np.random.normal(loc=0, scale=in_dimension, size=out_dimension)

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


class Conv2d(NetworkModuleWithParams):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.init_weights(in_channels, out_channels, kernel_size)
        self.stride = stride
        self.padding = padding
        self.x_cols = None
        self.a_minus_1 = None

    def init_weights(self, in_channels, out_channels, kernel_size):
        self.weights = np.random.normal(loc=0, scale=1, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.random.normal(loc=0, scale=1, size=out_channels)

    def __call__(self, input_):
        self.layer_input = input_
        self.layer_output, self.x_cols = util_functions.conv2d(input_, self.weights, self.biases, self.stride)
        return self.layer_output

    def backward(self, next_layer_grad):
        self.b_grad = np.sum(next_layer_grad, axis=(2, 3))

        N, C, H, W = self.layer_input.shape
        D, _, h, w = self.weights.shape

        next_layer_grad_reshaped = next_layer_grad.transpose(1, 2, 3, 0).reshape(D, -1)
        # self.w_grad = (next_layer_grad_reshaped @ self.x_cols.T).reshape(self.weights.shape)

        dx_cols = self.weights.reshape(D, -1).T @ next_layer_grad_reshaped
        self.layer_grad = util_functions.col2im_indices(dx_cols, self.layer_input.shape, h, w, self.padding, self.stride)

        w_grad = np.zeros((N, D, C, h, w))
        for n in range(N):
            for d in range(D):
                for c in range(C):
                    layer_grad = next_layer_grad[n, d, :, :]
                    layer_grad_h, layer_grad_w = layer_grad.shape
                    layer_grad = layer_grad.reshape(1, 1, layer_grad_h, layer_grad_w)

                    layer_input = self.layer_input[n, c, :, :]
                    layer_input_h, layer_input_w = layer_input.shape
                    layer_input = layer_input.reshape(1, 1, layer_input_h, layer_input_w)

                    biases = np.zeros(1)
                    filter_grad, _ = util_functions.conv2d(layer_input, layer_grad, biases, stride=self.stride, padding=self.padding)
                    w_grad[n, d, c, :, :] = filter_grad

        self.w_grad = w_grad
        return self.layer_grad


class Linear(NetworkModuleWithParams):

    def __init__(self, in_dimension, out_dimension):
        super(Linear, self).__init__()
        self.init_weights(in_dimension, out_dimension)

    def init_weights(self, in_dimension, out_dimension):
        self.weights = np.random.normal(loc=0, scale=1, size=(out_dimension, in_dimension))
        self.biases = np.random.normal(loc=0, scale=1, size=out_dimension)

    def __call__(self, input_):
        self.layer_input = input_
        self.layer_output = util_functions.linear(input_, self.weights, self.biases)
        return self.layer_output

    def backward(self, next_layer_grad):
        self.layer_grad = next_layer_grad @ self.weights

        batch_size, _ = self.layer_grad.shape
        self.w_grad = next_layer_grad.reshape(batch_size, -1, 1) @ self.layer_input.reshape(batch_size, 1, -1)
        self.b_grad = next_layer_grad
        return self.layer_grad


class Relu(NetworkModule):

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = util_functions.relu(z)
        return self.layer_output

    def backward(self, next_layer_gard):
        self.layer_grad = next_layer_gard * util_functions.relu_derivative(self.layer_input)
        return self.layer_grad


class Softmax(NetworkModule):

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = util_functions.softmax(z)
        return self.layer_output

    def backward(self, label):
        # assuming cross entropy loss
        self.layer_grad = self.layer_output - label
        return self.layer_grad


class Sigmoid(NetworkModule):

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = util_functions.sigmoid(z)
        return self.layer_output

    def backward(self, next_layer_grad):
        self.layer_grad = next_layer_grad * util_functions.sigmoid_derivative(self.layer_input)
        return self.layer_grad


class Flatten(NetworkModule):

    def __call__(self, x):
        self.layer_input = x
        batch_size = x.shape[0]
        self.layer_output = np.reshape(x, (batch_size, -1))
        return self.layer_output

    def backward(self, next_layer_grad):
        N, C, H, W = self.layer_input.shape
        self.layer_grad = np.reshape(next_layer_grad, (N, C, H, W))
        return self.layer_grad


class Dropout(NetworkModule):

    def __init__(self, rate, mode="train"):
        super(Dropout, self).__init__()
        self.rate = rate
        self.mode = mode
        self.dropout_mask = None

    def get_mask(self, shape):
        drop_probabilities = np.random.rand(*shape)
        drop_mask = np.where(drop_probabilities <= self.rate, 0, 1)
        self.dropout_mask = drop_mask.reshape(shape)

        return self.dropout_mask

    def __call__(self, z):
        self.layer_input = z

        if self.mode == 'train':
            self.dropout_mask = self.get_mask(self.layer_input.shape)
            self.layer_output = self.layer_input * self.dropout_mask

        else:
            self.layer_output = self.layer_input * (1 - self.rate)

        return self.layer_output

    def backward(self, next_layer_grad):
        self.layer_grad = next_layer_grad * self.dropout_mask
        return self.layer_grad
