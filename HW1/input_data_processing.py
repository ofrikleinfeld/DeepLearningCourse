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


def load_dataset(data_file, labeled=True, normalize=False):
    data_samples = []
    labels = []
    outputs = []
    with open(data_file, "r") as f:
        for line in f:
            data_with_labels = line[:-1].split(",")
            label, data = data_with_labels[0],  [float(p) for p in data_with_labels[1:]]

            # transform from 1 based index to 0 based index
            # then perform one hot encoding
            if labeled:
                if label != '?':
                    label_index = int(float(label)) - 1
                    one_hot_label = vectorized_result(label_index)
                    labels.append(one_hot_label)

            data_samples.append(data)

    data_array = np.array(data_samples).reshape((len(data_samples), -1, 1))
    if normalize == True:
        data_mean = data_array.mean(axis=0)
        data_std = data_array.std(axis=0)
        data_array = (data_array - data_mean) / data_std
        outputs += [data_mean, data_std]
    elif (type(normalize) == tuple):
        data_array = (data_array - normalize[0]) / normalize[1]
    outputs.append(data_array)
    if labeled:
        labels_array = np.array(labels)
        outputs.append(labels_array)
    return outputs


def save_data_as_pickle_gz(data, file_name):
    with gzip.open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_training_validation_data(path):
    with gzip.open(path, "rb") as f:
        training_data, training_labels, validation_data, validation_labels = pickle.load(f)

    return training_data, training_labels, validation_data, validation_labels


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def crop_dataset(data, desired_dim=28):
    batch_size, flatten_size, _ = data.shape
    num_channels = 3
    image_size = int(np.sqrt(flatten_size / num_channels))
    data_original_shape = data.reshape((batch_size, num_channels, image_size, image_size))
    data_cropped = data_original_shape[:, :, :desired_dim, :desired_dim]
    data_cropped_flatten = data_cropped.reshape((batch_size, -1, 1))

    return data_cropped_flatten
