import random

from data_processing import load
from Models import FeedForwardNet
from Optimizers import SGDOptimizer

random.seed(123)

train_date, test_data = load(train_size=50000, test_size=10000)
net = FeedForwardNet([784, 100, 50, 30, 20, 10])
optimizer = SGDOptimizer(lr=0.01)
n_epochs = 10
batch_size = 32

# split to batches and feed to model
for e in range(n_epochs):

    random.shuffle(list(train_date))
    batches = [train_date[k:k + batch_size] for k in range(0, len(train_date), batch_size)]
    for batch in batches:
        x_batch, y_batch = batch

        # forward + backward pass
        net.forward_pass(x_batch)
        dw, db = net.compute_gradients(y_batch)
        # update weight after SGD step
        new_w, new_b = optimizer.make_step(net.weights, net.biases, dw, db, len(x_batch))
        net.set_parameters(new_w, new_b)
        net.zero_gradients()

    