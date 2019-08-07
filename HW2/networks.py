import pickle

import modules as nn


class NN(object):
    def __init__(self):
        self.layers = None
        self.mode = 'train'

    def __call__(self, x):
        for l in self.layers:
            if isinstance(l, nn.NetworkModuleWithMode):
                x = l(x, mode=self.mode)
            else:
                x = l(x)
        return x

    def __str__(self):
        return " ".join([l.__str__() for l in self.layers])

    def backward(self, labels):
        last_grad = self.layers[-1].backward(labels)
        for l in reversed(self.layers[:-1]):
            last_grad = l.backward(last_grad)

    def set_mode(self, mode):
        if mode not in (['train', 'test']):
            raise ValueError('mode should be either train or test')
        self.mode = mode

    def set_forward(self):
        raise NotImplementedError("Sub class must implement forward pass")

    def save_model(self, path="trained_model.pkl"):
        with open(path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(path="trained_model.pkl"):
        with open(path, "rb") as input_file:
            model = pickle.load(input_file)

        return model


class SimpleCNN(NN):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1)
        self.relu1 = nn.LeakyRelu()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=4, stride=1)
        self.relu2 = nn.LeakyRelu()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1)
        self.relu3 = nn.LeakyRelu()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_dimension=48, out_dimension=10)
        self.softmax = nn.Softmax()
        self.set_forward()

    def set_forward(self):
        self.layers = [self.conv1, self.relu1, self.max_pool1,
                       self.conv2, self.relu2, self.max_pool2,
                       self.conv3, self.relu3, self.max_pool3,
                       self.flatten, self.linear, self.softmax]


class SimplerCNN(NN):
    def __init__(self):
        super(SimplerCNN, self).__init__()
        self.dropout2d_input = nn.Dropout2d(rate=0.3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=3, padding=2)
        self.relu1 = nn.LeakyRelu()
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=3, stride=3, padding=3)
        self.relu2 = nn.LeakyRelu()
        self.dropout2d_conv1 = nn.Dropout2d(rate=0.5)
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=4)
        self.relu3 = nn.LeakyRelu()
        self.flatten = nn.Flatten()
        self.dropout2d_conv2 = nn.Dropout2d(rate=0.2)
        self.linear = nn.Linear(in_dimension=360, out_dimension=180)
        self.relu4 = nn.LeakyRelu()
        self.bn1 = nn.BatchNorm()
        self.dropout3 = nn.Dropout(rate=0.3)
        self.linear2 = nn.Linear(in_dimension=180, out_dimension=10)
        self.bn2 = nn.BatchNorm()
        self.softmax = nn.Softmax()
        self.set_forward()

    def set_forward(self):
        self.layers = [self.dropout2d_input, self.conv1, self.relu1,
                       self.conv2, self.relu2, self.dropout2d_conv1,
                       self.conv3, self.relu3, self.dropout2d_conv2,
                       self.flatten, self.linear,  self.relu4, self.bn1,
                       self.dropout3, self.linear2, self.bn2, self.softmax]


class FullyConnected(NN):

    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = nn.Linear(in_dimension=3072, out_dimension=256)
        self.relu1 = nn.LeakyRelu()
        self.dropout1 = nn.Dropout(rate=0.3)
        self.linear2 = nn.Linear(in_dimension=256, out_dimension=10)
        self.softmax = nn.Softmax()
        self.set_forward()

    def set_forward(self):
        self.layers = [self.linear1, self.relu1,
                       self.linear2, self.softmax]

