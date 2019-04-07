import numpy as np

from util_functions import relu, relu_derivative, softmax


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.z_values = [None] * self.num_layers
        self.activations = [None] * self.num_layers
        self.dw = [None] * (self.num_layers - 1)
        self.db = [None] * (self.num_layers - 1)

    def set_weights(self, new_weights):
        self.weights = new_weights

    def set_biases(self, new_biases):
        self.biases = new_biases

    def zero_gradients(self):
        self.z_values = [None] * self.num_layers
        self.activations = [None] * self.num_layers
        self.dw = [None] * (self.num_layers - 1)
        self.db = [None] * (self.num_layers - 1)

    def forward_pass(self, x_batch):
        self.z_values[0] = x_batch
        self.activations[0] = x_batch
        for l in range(self.num_layers - 1):
            self.z_values[l+1] = np.dot(self.weights[l], self.activations[l]) + self.biases[l]
            self.activations[l+1] = relu(self.z_values[l+1])

        return self.z_values[-1]

    def compute_gradients(self, y_batch):
        delta = [None] * (self.num_layers - 1)
        delta[-1] = softmax(self.z_values[-1]) - y_batch
        self.dw[-1] = np.dot(delta[-1], self.activations[-2].transpose())

        for l in range(2, self.num_layers):
            delta[-l] = np.dot(self.weights[-l+1].transpose(), delta[-l+1]) * relu_derivative(self.z_values[-l])
            self.dw[-l] = np.dot(delta[-l], self.activations[-l-1].transpose())

        self.db = delta
        return self.dw, self.db

    def predict_batch(self, x_batch):
        # self.zero_gradients()
        # return np.argmax(softmax(self.forward_pass(x_batch)), axis=1)
        pass