import datetime
import numpy as np

from input_data_processing import load_dataset, save_data_as_pickle_gz, load_training_validation_data
from util_functions import shuffle_dataset, plot_accuracy_graph, write_output_to_log
from Models import FeedForwardNet, DropoutFeedForwardNet
from Optimizers import SGDOptimizer

np.random.seed(123)

# train_data, train_labels = load_dataset("data/train.csv")
# validation_data, validation_labels = load_dataset("data/validate.csv")
# test_data = load_dataset("data/test.csv", labeled=False)
# save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels), file_name="data/training_validation.pkl.gz")
# save_data_as_pickle_gz(test_data, file_name="data/test.pkl.gz")

train_data, train_labels, validation_data, validation_labels = load_training_validation_data()

net = FeedForwardNet([3072, 200, 10])
optimizer = SGDOptimizer(lr=0.01)
# net = DropoutFeedForwardNet(sizes=[784, 40, 10], dropout_rate=0.5)
# optimizer = SGDOptimizer(lr=0.01, weights_decay='L2', weights_decay_rate=0.0001)
n_epochs = 50
batch_size = 1
train_accuracy = []
validation_accuracy = []

# write to log file
file_path = "log/training_output_{0}.txt".format(datetime.datetime.now())
f = open(file_path, "w")

write_output_to_log(f, "Model Type: {0}\n".format(type(net)))
write_output_to_log(f, "Network architecture: {0}\n".format(net.sizes))
write_output_to_log(f, "Learning rate: {0}\n".format(optimizer.lr))
write_output_to_log(f, "Weights decay and rate: {0}, {1}\n".format(optimizer.weights_decay, optimizer.weights_decay_rate))
write_output_to_log(f, "Number of epochs: {0}\n".format(n_epochs))
write_output_to_log(f, "Batch size: {0}\n".format(batch_size))

# split to batches and feed to model
for e in range(n_epochs):

    train_data, train_labels = shuffle_dataset(train_data, train_labels)
    batch_indices = range(0, len(train_data), batch_size)
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

    write_output_to_log(f, "Epoch {0}: Train epoch accuracy is {1}\n".format(e+1, train_epoch_accuracy))
    write_output_to_log(f, "Epoch {0}: Validation epoch accuracy is {1}\n".format(e+1, validation_epoch_accuracy))

write_output_to_log(f, "Training ended at: {0}\n".format(datetime.datetime.now()))
f.close()
plot_accuracy_graph(train_accuracy, validation_accuracy)
