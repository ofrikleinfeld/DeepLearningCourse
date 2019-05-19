import datetime
import numpy as np

from input_data_processing import load_dataset, save_data_as_pickle_gz, load_training_validation_data
from crop_and_augment import crop_and_augment
from util_functions import shuffle_dataset, write_output_to_log
from Models import DropoutFeedForwardNet
from Optimizers import SGDOptimizer


np.random.seed(1312)

crop_and_augment()

train_mean, train_std, train_data, train_labels = load_dataset("data/train_cropped_smaller.csv", normalize=True)
validation_data, validation_labels = load_dataset("data/valid_cropped_smaller.csv", normalize=(train_mean, train_std))
test_data = load_dataset("data/test_cropped_smaller.csv", labeled=True, normalize=(train_mean, train_std))
save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels),
                      file_name="data/training_cropped_smaller_validation_normalized.pkl.gz")
save_data_as_pickle_gz(test_data, file_name="data/test_normalized_cropped_smaller.pkl.gz")

train_mean, train_std, train_data, train_labels = load_dataset("data/train_mirrored_cropped_smaller.csv", normalize=True)
validation_data, validation_labels = load_dataset("data/valid_cropped_smaller.csv", normalize=(train_mean, train_std))
save_data_as_pickle_gz((train_data, train_labels, validation_data, validation_labels),
                      file_name="data/training_mirrored_cropped_smaller_validation_normalized.pkl.gz")

train_data, train_labels, validation_data, validation_labels = load_training_validation_data('data/training_cropped_smaller_validation_normalized.pkl.gz')
train_data_mirror, train_labels_mirror, _, _ = load_training_validation_data('data/training_mirrored_cropped_smaller_validation_normalized.pkl.gz')
train_data = np.vstack((train_data, train_data_mirror))
train_labels = np.vstack((train_labels, train_labels_mirror))


net = DropoutFeedForwardNet(sizes=[2352, 256, 10], dropout_rates=[0.1, 0.5])
optimizer = SGDOptimizer(lr=0.9)
n_epochs = 500
batch_size = 128
train_accuracy = []
validation_accuracy = []
# load_weights = ('best_weights.npy', 'best_biases.npy')
# sw_threshold = 0.479

# write to log file
file_path = "log/training_output_{0}.txt".format(datetime.datetime.now())
f = open(file_path, "w")

write_output_to_log(f, "Model Type: {0}\n".format(type(net)))
write_output_to_log(f, "Network architecture: {0}\n".format(net.sizes))
write_output_to_log(f, "Learning rate: {0}\n".format(optimizer.lr))
write_output_to_log(f, "Weights decay and rate: {0}, {1}\n".format(optimizer.weights_decay, optimizer.weights_decay_rate))
write_output_to_log(f, "Number of epochs: {0}\n".format(n_epochs))
write_output_to_log(f, "Batch size: {0}\n".format(batch_size))

ni_cnt = 0


train_mean, train_std, _, _ = load_dataset("data/train_cropped_smaller.csv", normalize=True)
test_data, _ = load_dataset("data/test_cropped_smaller.csv", labeled=True, normalize=(train_mean, train_std))


# if load_weights:
#     best_w = np.load(load_weights[0])
#     best_biases = np.load(load_weights[1])
#     net.set_parameters(best_w, best_biases)


# split to batches and feed to model
for e in range(n_epochs):

    train_data, train_labels = shuffle_dataset(train_data, train_labels)
    train_data_samp = train_data[:10000, :, :]
    train_labels_samp = train_labels[:10000, :, :]
    batch_indices = range(0, round((len(train_data_samp)/batch_size)) * batch_size, batch_size)
    for k in batch_indices:
        x_batch = train_data_samp[k: k + batch_size]
        y_batch = train_labels[k: k + batch_size]

        # forward + backward pass
        net.forward_pass(x_batch)
        dw, db = net.compute_gradients(y_batch)
        # update weight after SGD step
        new_w, new_b = optimizer.make_step(net.weights, net.biases, dw, db, len(x_batch))
        net.set_parameters(new_w, new_b)
        net.zero_gradients()

    # compute epoch accuracy for train and validation data
    train_batch_indices = range(0, len(train_data_samp), batch_size)
    validation_batch_indices = range(0, len(validation_data), batch_size)
    train_num_correct = 0
    for k in train_batch_indices:
        x_batch = train_data_samp[k: k + batch_size]
        y_batch = train_labels_samp[k: k + batch_size]
        batch_predictions = net.predict_batch(x_batch)
        train_num_correct += net.get_num_correct_predictions(batch_predictions, y_batch)

    validation_num_correct = 0
    for k in validation_batch_indices:
        x_batch = validation_data[k: k + batch_size]
        y_batch = validation_labels[k: k + batch_size]
        batch_predictions = net.predict_batch(x_batch)
        validation_num_correct += net.get_num_correct_predictions(batch_predictions, y_batch)

    train_epoch_accuracy = train_num_correct / len(train_data_samp)
    validation_epoch_accuracy = validation_num_correct / len(validation_data)
    if validation_accuracy and validation_epoch_accuracy > max(validation_accuracy) \
        and validation_epoch_accuracy > sw_threshold:
        write_output_to_log(f, "Best validation accuracy %.2f so far. Saving weights"
                            % validation_epoch_accuracy)
        np.save('best_weights', new_w)
        np.save('best_biases', new_b)
        ni_cnt = 0
    elif validation_accuracy and validation_epoch_accuracy < max(validation_accuracy):
        ni_cnt += 1
        if ni_cnt > 8:
            optimizer.lr = optimizer.lr * 0.95
            write_output_to_log(f, "No improvement. lr decreased to %.4f"
                                % optimizer.lr)
            ni_cnt = 4
    else:
        ni_cnt = 0
    train_accuracy.append(train_epoch_accuracy)
    validation_accuracy.append(validation_epoch_accuracy)

    write_output_to_log(f, "Epoch {0}: Train epoch accuracy is {1}\n".format(e+1, train_epoch_accuracy))
    write_output_to_log(f, "Epoch {0}: Validation epoch accuracy is {1}\n".format(e+1, validation_epoch_accuracy))

write_output_to_log(f, "Training ended at: {0}\n".format(datetime.datetime.now()))
f.close()


# predict on test set and write to output file
test_predictions = []
for test_sample in test_data:
    prediction_index = net.predict_batch(test_sample.reshape(1, -1, 1))
    prediction = prediction_index[0][0] + 1
    test_predictions.append(prediction)

with open("output.txt", "w") as f:
    for pred in test_predictions:
        f.write(str(pred) + "\n")

