import pickle
import gzip

import numpy as np


def load():
    """Return a tuple containing ``(training_data, test_data)``.
    In particular, ``train_from_file`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``test_from_file`` is a list containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Hence, we're using different label formats for
    the training data and the test data.  """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    train_from_file, validation_from_file, test_from_file = pickle.load(f, encoding='iso-8859-1')
    f.close()
    training_inputs = np.array([np.reshape(x, (784, 1)) for x in train_from_file[0]])
    training_labels = np.array([vectorized_result(y) for y in train_from_file[1]])
    validation_inputs = np.array([np.reshape(x, (784, 1)) for x in validation_from_file[0]])
    validation_labels = np.array([vectorized_result(y) for y in validation_from_file[1]])
    test_inputs = np.array([np.reshape(x, (784, 1)) for x in test_from_file[0]])
    test_labels = np.array([vectorized_result(y) for y in test_from_file[1]])
    return training_inputs, training_labels, validation_inputs, validation_labels, test_inputs, test_labels


def load_dataset(data_file, labeled=True):
    data_samples = []
    labels = []
    with open(data_file, "r") as f:
        for line in f:
            data_with_labels = line[:-1].split(",")
            label, data = data_with_labels[0],  [float(p) for p in data_with_labels[1:]]

            # transform from 1 based index to 0 based index
            # then perform one hot encoding
            if labeled:
                label_index = int(label) - 1
                one_hot_label = vectorized_result(label_index)
                labels.append(one_hot_label)

            data_samples.append(data)

    data_array = np.array(data_samples).reshape((len(data_samples), -1, 1))
    if labeled:
        labels_array = np.array(labels)
        return data_array, labels_array
    else:
        return data_array


def save_data_as_pickle_gz(data, file_name):
    with gzip.open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_training_validation_data():
    with gzip.open("data/training_validation.pkl.gz", "rb") as f:
        training_data, training_labels, validation_data, validation_labels = pickle.load(f)

    return training_data, training_labels, validation_data, validation_labels


def load_test_date():
    with gzip.open("data/test.pkl.gz", "rb") as f:
        test_data = pickle.load(f)

    return test_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def crop_dataset(data, desired_dim=28):
    dataset_length, features_flatten_size, _ = data.shape
    num_channels = 3
    image_size = int(np.sqrt(features_flatten_size / num_channels))

    total_crop_size = image_size - desired_dim
    edges_crop_size = int(total_crop_size / 2)
    start_edge = edges_crop_size
    end_edge = image_size - edges_crop_size

    data_original_shape = data.reshape((dataset_length, num_channels, image_size, image_size))
    data_cropped = data_original_shape[:, :, start_edge:end_edge, start_edge:end_edge]
    data_cropped_flatten = data_cropped.reshape((dataset_length, -1, 1))

    return data_cropped_flatten
