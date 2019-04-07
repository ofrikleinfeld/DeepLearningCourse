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
    training_inputs = np.array([np.reshape(x, 784) for x in train_from_file[0]])
    training_labels = np.array([vectorized_result(y) for y in train_from_file[1]])
    validation_inputs = np.array([np.reshape(x, 784) for x in validation_from_file[0]])
    validation_labels = np.array([vectorized_result(y) for y in validation_from_file[1]])
    test_inputs = np.array([np.reshape(x, 784) for x in test_from_file[0]])
    test_labels = np.array([vectorized_result(y) for y in test_from_file[1]])
    return training_inputs, training_labels, validation_inputs, validation_labels, test_inputs, test_labels


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros(10)
    e[j] = 1.0
    return e
