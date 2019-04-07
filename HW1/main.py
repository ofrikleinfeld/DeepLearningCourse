import random
import numpy as np

from input_data_processing import load
from util_functions import shuffle_dataset, plot_accuracy_graph
from Models import FeedForwardNet
from Optimizers import SGDOptimizer

random.seed(123)

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load()
net = FeedForwardNet([784, 50, 20, 10])
optimizer = SGDOptimizer(lr=0.01, weights_decay='L2', weights_decay_rate=0.0001)
n_epochs = 15
batch_size = 32
train_accuracy = []
validation_accuracy = []

# split to batches and feed to model
for e in range(n_epochs):

    train_data, train_labels = shuffle_dataset(train_data, train_labels)
    batch_indices = range(0, len(train_data), batch_size)
    # batch_indices = range(0, 33, batch_size)
    for k in batch_indices:
        x_batch = train_data[k: k + batch_size]
        y_batch = train_labels[k: k + batch_size]

        # forward + backward pass
        net.forward_pass(x_batch)
        dw, db = net.compute_gradients(y_batch)
        # update weight after SGD step
        new_w, new_b = optimizer.make_step(net.weights, net.biases, dw, db, len(x_batch))
        net.set_parameters(new_w, new_b)
        net.zero_gradients()

    # compute epoch accuracy for train and validation data
    train_batch_indices = range(0, len(train_data), batch_size)
    validation_batch_indices = range(0, len(validation_data), batch_size)

    train_num_correct = 0
    for k in train_batch_indices:
        x_batch = train_data[k: k + batch_size]
        y_batch = train_labels[k: k + batch_size]
        batch_predictions = net.predict_batch(x_batch)
        train_num_correct += net.get_num_correct_predictions(batch_predictions, y_batch)

    validation_num_correct = 0
    for k in validation_batch_indices:
        x_batch = validation_data[k: k + batch_size]
        y_batch = validation_labels[k: k + batch_size]
        batch_predictions = net.predict_batch(x_batch)
        validation_num_correct += net.get_num_correct_predictions(batch_predictions, y_batch)

    train_epoch_accuracy = train_num_correct / len(train_data)
    validation_epoch_accuracy = validation_num_correct / len(validation_data)
    train_accuracy.append(train_epoch_accuracy)
    validation_accuracy.append(validation_epoch_accuracy)

    print("Epoch {0}: Train epoch accuracy is {1}".format(e+1, train_epoch_accuracy))
    print("Epoch {0}: Validation epoch accuracy is {1}".format(e+1, validation_epoch_accuracy))

plot_accuracy_graph(train_accuracy, validation_accuracy)