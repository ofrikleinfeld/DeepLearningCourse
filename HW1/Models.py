import numpy as np

from util_functions import relu, relu_derivative, softmax, sigmoid, sigmoid_derivative


class FeedForwardNet(object):

    @staticmethod
    def get_num_correct_predictions(predictions, labels):
        correct_label = np.argmax(labels, axis=1)
        return np.sum(predictions == correct_label)

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.z_values = [None] * self.num_layers
        self.activations = [None] * self.num_layers
        self.dw = [None] * (self.num_layers - 1)
        self.db = [None] * (self.num_layers - 1)

    def set_parameters(self, new_weights, new_biases):
        self.weights = new_weights
        self.biases = new_biases

    def zero_gradients(self):
        self.z_values = [None] * self.num_layers
        self.activations = [None] * self.num_layers
        self.dw = [None] * (self.num_layers - 1)
        self.db = [None] * (self.num_layers - 1)

    def forward_pass(self, x_batch):
        self.z_values[0] = x_batch
        self.activations[0] = x_batch

        # feed forward until one layer before final layer
        for l in range(self.num_layers - 2):
            self.z_values[l+1] = self.activations[l] @ self.weights[l].transpose() + self.biases[l]
            self.activations[l+1] = sigmoid(self.z_values[l+1])

        # for final layer, activation is softmax. Will be performed during backprop
        self.z_values[-1] = self.activations[-2] @ self.weights[-1].transpose() + self.biases[-1]

        return self.z_values[-1]

    def compute_gradients(self, y_batch):
        delta = [None] * (self.num_layers - 1)
        delta[-1] = softmax(self.z_values[-1]) - y_batch
        self.dw[-1] = delta[-1].reshape(-1, delta[-1].shape[1], 1) @ self.activations[-2].reshape(-1, 1, self.activations[-2].shape[1])

        for l in range(2, self.num_layers):
            delta[-l] = delta[-l+1] @ self.weights[-l+1] * sigmoid_derivative(self.z_values[-l])
            self.dw[-l] = delta[-l].reshape(-1, delta[-l].shape[1], 1) @ self.activations[-l-1].reshape(-1, 1, self.activations[-l-1].shape[1])

        self.db = delta
        return self.dw, self.db

    def predict_batch(self, x_batch):
        a = x_batch
        for l in range(self.num_layers - 2):
            z = a @ self.weights[l].transpose() + self.biases[l]
            a = sigmoid(z)

        output = a @ self.weights[-1].transpose() + self.biases[-1]
        return np.argmax(softmax(output), axis=1)

