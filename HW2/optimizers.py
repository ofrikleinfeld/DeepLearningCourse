import numpy as np
import modules as nn


class SGDOptimizer(object):
    def __init__(self, model, lr, weights_decay=None, weights_decay_rate=None, momentum=None):
        self.model = model
        self.lr = lr
        self.weights_decay = weights_decay
        self.weights_decay_rate = weights_decay_rate
        self.m = None
        if momentum:
            self.m = momentum

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
                if self.m:
                    vw, vb = l.vw, l.vb
                    if isinstance(l, nn.Conv2d):
                        vw = self.m * vw + self.lr * weights_grad
                    else:
                        vw = self.m * vw + self.lr * np.sum(weights_grad, axis=0)
                    vb = self.m * vb + self.lr * np.sum(biases_grad, axis=0)
                    w -= vw
                    b -= vb
                    l.set_vw(vw)
                    l.set_vb(vb)
                else:
                    if isinstance(l, nn.Conv2d):
                        w -= self.lr * weights_grad
                    else:
                        w -= self.lr * np.sum(weights_grad, axis=0)
                    b -= self.lr * np.sum(biases_grad, axis=0)
                # update the layer weights after SGD step
                l.set_weights(w)
                l.set_biases(b)
            elif isinstance(l, nn.BatchNorm):
                gamma, beta = l.gamma, l.beta
                gamma_grad, beta_grad = l.dgamma, l.dbeta
                if self.m:
                    vgamma, vbeta = l.vgamma, l.vbeta
                    vgamma = self.m * vgamma + self.lr * np.sum(gamma_grad, axis=0)
                    vbeta = self.m * vbeta + self.lr * np.sum(beta_grad, axis=0)
                    gamma -= vgamma
                    beta -= vbeta
                    l.set_vgamma(vgamma)
                    l.set_vbeta(vbeta)
                else:
                    gamma -= self.lr * np.sum(gamma_grad, axis=0)
                    beta -= self.lr * np.sum(beta_grad, axis=0)
                l.set_gamma(gamma)
                l.set_beta(beta)
