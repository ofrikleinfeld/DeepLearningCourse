import numpy as np
import modules as nn


class SGDOptimizer(object):
    def __init__(self, model, lr, weights_decay=None, weights_decay_rate=None):
        self.model = model
        self.lr = lr
        self.weights_decay = weights_decay
        self.weights_decay_rate = weights_decay_rate

    def make_step(self):
        layers = self.model.layers
        for l in layers:
            if isinstance(l, nn.NetworkModuleWithParams):
                w, b = l.weights, l.biases
                weights_grad, biases_grad = l.w_grad,  l.b_grad

                # perform weights decay
                if self.weights_decay is not None:
                    if self.weights_decay == "L1":
                        w -= self.weights_decay_rate
                        b -= self.weights_decay_rate

                    elif self.weights_decay == "L2":
                        w -= self.weights_decay_rate * w
                        b -= self.weights_decay_rate * b

                # perform SGD step with average batch gradient
                w -= self.lr * np.mean(weights_grad, axis=0)
                b -= self.lr * np.mean(biases_grad, axis=0)

                # update the layer weights after SGD step
                l.set_weights(w)
                l.set_biases(b)
