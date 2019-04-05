import numpy as np


class SGDOptimizer(object):
    def __init__(self, lr, batch_size, weights_decay=None, weights_decay_rate=None):
        self.lr = lr
        self.batch_size = batch_size
        self.weights_decay = weights_decay
        self.weights_decay_rate = weights_decay_rate

    def make_step(self, weights, biases, dw, db):
        # perform weights decay
        if self.weights_decay is not None:
            if self.weights_decay == 'L1':
                weights = [w - self.weights_decay_rate for w in weights]
                biases = [b - self.weights_decay_rate for b in biases]

            elif self.weights_decay == 'L2':
                weights = [w - self.weights_decay_rate * w for w in weights]
                biases = [b - self.weights_decay_rate * b for b in biases]

        # perform SGD step with average batch gradient
        new_weights = [w - self.lr * np.sum(dw) / self.batch_size for w, dw in zip(weights, dw)]
        new_biases = [b - self.lr * np.sum(db) / self.batch_size for b, db in zip(biases, db)]

        return new_weights, new_biases

