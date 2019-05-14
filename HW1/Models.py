from util_functions import relu, relu_derivative, softmax, sigmoid, sigmoid_derivative

import pickle
import gzip
import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


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

    def save_model_to_pickle(self):
        if not os.path.exists("models"):
            os.mkdir("models")

        class_name = self.__class__.__name__
        file_name = "{0}.pkl.gz".format(class_name)
        file_path = os.path.join("models", file_name)
        with gzip.open(file_path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model_from_pickle(file_path):
        with gzip.open(file_path, "rb") as f:
            return pickle.load(f)


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


class KerasDropoutFeedForwardNet(object):

    def __init__(self, sizes, dropout_rate):
        self.sizes = sizes
        self.dropout_rate = dropout_rate
        self.model = Sequential()
        self._define_network()

    def _define_network(self):
        input_size = self.sizes[0]
        first_hidden_size = self.sizes[1]
        output_size = self.sizes[-1]

        self.model.add(Dense(first_hidden_size, activation='sigmoid', input_dim=input_size))
        self.model.add(Dropout(self.dropout_rate))

        for hidden_size in self.sizes[2:-1]:
            self.model.add(Dense(hidden_size, activation='sigmoid'))
            self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(output_size, activation='softmax'))

    def compile_model(self, lr=0.01, decay_rate=1e-6):
        sgd_optimizer = SGD(lr=lr, decay=decay_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])

    def train_model(self, x_train, y_train, x_validation, y_validation, epochs=5, batch_size=32):
        print(self.model.summary())
        self.model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs, batch_size=batch_size)

