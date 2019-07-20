import numpy as np
import util_functions as util_functions


class NetworkModule(object):

    def __init__(self):
        self.layer_input = None
        self.layer_output = None
        self.layer_grad = None

    def __call__(self, input_):
        raise NotImplementedError("Sub class must implement forward pass")

    def backward(self, next_layer_grad):
        raise NotImplementedError("Sub class must implement backward pass")


class NetworkModuleWithMode(NetworkModule):

    def __init__(self):
        super(NetworkModuleWithMode, self).__init__()

    def __call__(self, input_, mode):
        raise NotImplementedError("Sub class must implement forward pass")


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
        if(len(new_biases.shape) != 2):
            self.biases = new_biases.reshape(new_biases.shape[0], 1)
        else:
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
        self.cache = None

    def init_weights(self, in_channels, out_channels, kernel_size):
        self.weights = np.random.normal(loc=0, scale=1, size=(out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.random.normal(loc=0, scale=1, size=(out_channels, 1))

    def __call__(self, input_):
        self.layer_input = input_
        n_filters, d_filter, h_filter, w_filter = self.weights.shape
        n_x, d_x, h_x, w_x = input_.shape
        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)

        X_col = util_functions.im2col_indices(input_, h_filter, w_filter, padding=self.padding, stride=self.stride)
        W_col = self.weights.reshape(n_filters, -1)

        out = W_col @ X_col + self.biases
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        self.x_cols = X_col
        self.cache = (input_, self.weights, self.biases, self.stride, self.padding, X_col)

        return out

    def backward(self, next_layer_grad):
        X, W, b, stride, padding, X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(next_layer_grad, axis=(2, 3))
        db = db.reshape(next_layer_grad.shape[0],n_filter, -1)

        dout_reshaped = next_layer_grad.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = util_functions.col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)
        self.w_grad = dW
        self.layer_grad = dX
        self.b_grad = db
        return self.layer_grad


class Linear(NetworkModuleWithParams):

    def __init__(self, in_dimension, out_dimension):
        super(Linear, self).__init__()
        self.init_weights(in_dimension, out_dimension)

    def init_weights(self, in_dimension, out_dimension):
        self.weights = np.random.normal(loc=0, scale=1, size=(out_dimension, in_dimension))
        self.biases = np.random.normal(loc=0, scale=1, size=(out_dimension, 1))

    def __call__(self, input_):
        self.layer_input = input_
        self.layer_output = util_functions.linear(input_, self.weights, self.biases)
        return self.layer_output

    def backward(self, next_layer_grad):
        self.layer_grad = next_layer_grad @ self.weights
        batch_size, _ = self.layer_grad.shape
        self.w_grad = next_layer_grad.reshape(batch_size, -1, 1) @ self.layer_input.reshape(batch_size, 1, -1)
        self.b_grad = next_layer_grad.reshape(batch_size, next_layer_grad.shape[1], 1)
        return self.layer_grad


class Relu(NetworkModule):

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = util_functions.relu(z)
        return self.layer_output

    def backward(self, next_layer_gard):
        self.layer_grad = next_layer_gard * util_functions.relu_derivative(self.layer_input)
        return self.layer_grad


class LeakyRelu(NetworkModule):

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = util_functions.leakyrelu(z)
        return self.layer_output

    def backward(self, next_layer_gard):
        self.layer_grad = next_layer_gard * util_functions.leakyrelu_derivative(self.layer_input)
        return self.layer_grad


class Tanh(NetworkModule):

    def __call__(self, z):
        self.layer_input = z
        self.layer_output = util_functions.tanh(z)
        return self.layer_output

    def backward(self, next_layer_gard):
        self.layer_grad = next_layer_gard * util_functions.tanh_derivative(self.layer_input)
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


class MaxPool2d(NetworkModule):

    def __init__(self, kernel_size, stride=2):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_cols = None
        self.max_idx = None

    def __call__(self, x):
        self.layer_input = x
        self.layer_output, self.max_idx, self.x_cols = \
            util_functions.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        return self.layer_output

    def backward(self, next_layer_grad):
        N, C, H, W = self.layer_input.shape

        dx_col = np.zeros_like(self.x_cols)
        next_layer_grad_col = next_layer_grad.transpose(2, 3, 0, 1).ravel()

        dx_col[self.max_idx, range(next_layer_grad_col.size)] = next_layer_grad_col
        layer_grad_unshaped = util_functions.col2im_indices(dx_col, (N * C, 1, H, W),
                                                            self.kernel_size, self.kernel_size, padding=0, stride=self.stride)

        self.layer_grad = layer_grad_unshaped.reshape(self.layer_input.shape)
        return self.layer_grad


class Dropout(NetworkModuleWithMode):

    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate
        self.dropout_mask = None

    def get_mask(self, shape):
        drop_probabilities = np.random.rand(*shape)
        drop_mask = np.where(drop_probabilities <= self.rate, 0, 1)
        self.dropout_mask = drop_mask.reshape(shape)

        return self.dropout_mask

    def __call__(self, z, mode):
        self.layer_input = z

        if mode == 'train':
            self.dropout_mask = self.get_mask(self.layer_input.shape)
            self.layer_output = self.layer_input * self.dropout_mask

        else:
            self.layer_output = self.layer_input * (1 - self.rate)

        return self.layer_output

    def backward(self, next_layer_grad):
        self.layer_grad = next_layer_grad * self.dropout_mask
        return self.layer_grad


class BatchNorm(NetworkModuleWithMode):

    def __init__(self):
        super(BatchNorm, self).__init__()
        self.gamma = np.random.rand()
        self.beta = np.random.rand()
        self.dgamma = None
        self.dbeta = None
        self.cache = None
        self.layer_grad = None
        self.bn_mean = 0
        self.bn_var = 0

    def __call__(self, z, mode):
        if mode == 'train':
            mu = np.mean(z, axis=0)
            var = np.var(z, axis=0)
            X_norm = (z - mu) / np.sqrt(var + 1e-8)
            out = self.gamma * X_norm + self.beta
            self.cache = (z, X_norm, mu, var,
                    self.gamma, self.beta)
            self.bn_mean = 0.9 * self.bn_mean + 0.1 * mu
            self.bn_var = 0.9 * self.bn_var + 0.1 * var
        else:
            out = (z - self.bn_mean) / np.sqrt(self.bn_var + 1e-8)
            out = self.gamma * out + self.beta
        return out

    def backward(self, next_layer_grad):
        X, X_norm, mu, var, gamma, beta = self.cache
        N, D = X.shape
        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)
        dX_norm = next_layer_grad * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)
        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        self.dgamma = np.sum(next_layer_grad * X_norm, axis=0)
        self.dbeta = np.sum(next_layer_grad, axis=0)
        self.layer_grad = dX
        return self.layer_grad

    def set_gamma(self, val):
        self.gamma = val

    def set_beta(self, val):
        self.beta = val
