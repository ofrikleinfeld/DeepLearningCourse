import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    # b = np.max(x, axis=1)[:, np.newaxis]
    # y = np.exp(x - b)
    # return y / np.sum(y, axis=1)[:, np.newaxis]
    z = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return z


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return relu(x) / np.abs(x)


def softmax(x):
    z = x - np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1)
    softmax = numerator / denominator[:, np.newaxis]
    return softmax


def shuffle_dataset(data, labels):
    assert len(data) == len(labels)
    random_indices = np.random.permutation(len(data))
    return data[random_indices], labels[random_indices]

def write_output_to_log(fp, message):
    fp.write(message)


def plot_accuracy_graph(train_accuracy, test_accuracy):
    epochs = range(1, len(train_accuracy)+1)
    plt.plot(epochs, train_accuracy, label='Training data')
    plt.plot(epochs, test_accuracy, label='Validation data')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Accuracy')
    plt.savefig('training_graph.pdf')
