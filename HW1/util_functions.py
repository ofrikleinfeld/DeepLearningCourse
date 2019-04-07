import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return relu(x) / np.abs(x)


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)


def shuffle_dataset(data, labels):
    assert len(data) == len(labels)
    random_indices = np.random.permutation(len(data))
    return data[random_indices], labels[random_indices]


def plot_accuracy_graph(train_accuracy, test_accuracy):
    epochs = range(1, len(train_accuracy)+1)
    plt.plot(epochs, train_accuracy, label='Training data')
    plt.plot(epochs, test_accuracy, label='Validation data')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Accuracy')
    plt.savefig('training_graph.pdf')
