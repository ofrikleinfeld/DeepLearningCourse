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
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
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
            self.z_values[l+1] = self.weights[l] @ self.activations[l] + self.biases[l]
            self.activations[l+1] = sigmoid(self.z_values[l+1])

        # for final layer, activation is softmax. Will be performed during backprop
        self.z_values[-1] = self.weights[-1] @ self.activations[-2] + self.biases[-1]

        return self.z_values[-1]

    def compute_gradients(self, y_batch):
        delta = [None] * (self.num_layers - 1)
        delta[-1] = softmax(self.z_values[-1]) - y_batch
        self.dw[-1] = delta[-1] @ self.activations[-2].transpose(0, 2, 1)

        for l in range(2, self.num_layers):
            delta[-l] = self.weights[-l+1].transpose() @ delta[-l+1] * sigmoid_derivative(self.z_values[-l])
            self.dw[-l] = delta[-l] @ self.activations[-l-1].transpose(0, 2, 1)

        self.db = delta
        return self.dw, self.db

    def predict_batch(self, x_batch):
        a = x_batch
        for l in range(self.num_layers - 2):
            z = self.weights[l] @ a + self.biases[l]
            a = sigmoid(z)

        output = self.weights[-1] @ a + self.biases[-1]
        return np.argmax(softmax(output), axis=1)


class DropoutFeedForwardNet(FeedForwardNet):

    def __init__(self, sizes, dropout_rate):
        super(DropoutFeedForwardNet, self).__init__(sizes)
        self.dropout_rate = dropout_rate

    def forward_pass(self, x_batch):
        self.z_values[0] = x_batch
        self.activations[0] = x_batch  # no dropout at input layer

        # feed forward until one layer before final layer
        for l in range(self.num_layers - 2):
            self.z_values[l + 1] = self.weights[l] @ self.activations[l] + self.biases[l]
            activations = sigmoid(self.z_values[l + 1])
            # add dropout layer
            dropout_layer = self.draw_dropout_layer(activations.shape)
            self.activations[l + 1] = activations * dropout_layer

        # for final layer, activation is softmax. Will be performed during backprop
        # no dropout for final (prediction) layer
        self.z_values[-1] = self.weights[-1] @ self.activations[-2] + self.biases[-1]

    def predict_batch(self, x_batch):
        a = x_batch
        for l in range(self.num_layers - 2):
            z = self.weights[l] @ a + self.biases[l]
            a = sigmoid(z) * (1 - self.dropout_rate)

        output = self.weights[-1] @ a + self.biases[-1]  # no dropout on last layer just softmax
        return np.argmax(softmax(output), axis=1)

    def draw_dropout_layer(self, shape):
        drop_probabilities = np.random.rand(*shape)
        dropout_layer = np.where(drop_probabilities <= self.dropout_rate, 0, 1)
        return dropout_layer
