import gzip
import pickle

import numpy as np


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

    data_array = np.array(data_samples).reshape((len(data_samples), -1))

    if normalize == True:
        data_mean = data_array.mean(axis=0)
        data_std = data_array.std(axis=0)
        data_array = (data_array - data_mean) / data_std
        outputs += [data_mean, data_std]

    elif type(normalize) == tuple:
        data_array = (data_array - normalize[0]) / normalize[1]
    outputs.append(data_array)

    if labeled:
        labels_array = np.array(labels)
        outputs.append(labels_array)

    return outputs


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros(10)
    e[j] = 1.0
    return e


def save_data_as_pickle_gz(data, file_name):
    with gzip.open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_training_validation_data(path):
    with gzip.open(path, "rb") as f:
        training_data, training_labels, validation_data, validation_labels = pickle.load(f)

    return training_data, training_labels, validation_data, validation_labels


def load_model_from_pickle(path):
    with gzip.open(path, "rb") as f:
        model = pickle.load(f)

    return model


def shuffle_batch(data, labels):
    assert len(data) == len(labels)
    random_indices = np.random.permutation(len(data))
    return data[random_indices], labels[random_indices]


def norm_img(img):
    img_tmp = img.reshape(3, 32,32)
    img_tmp = np.array([(samp - samp.mean()) /
                        (samp.std() + 1e-8) for samp in img_tmp])
    return img_tmp.reshape(3072)
