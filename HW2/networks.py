import modules as nn


class SimpleCNN(object):

    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1)
        self.relu1 = nn.Relu()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=4, stride=1)
        self.relu2 = nn.Relu()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1)
        self.relu3 = nn.Relu()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_dimension=48, out_dimension=10)
        self.softmax = nn.Softmax()
        self.layers = [self.conv1, self.relu1, self.max_pool1,
                        self.conv2, self.relu2, self.max_pool2,
                        self.conv3, self.relu3, self.max_pool3,
                        self.flatten, self.linear, self.softmax]

    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x

    def backward(self, labels):
        softmax_grad = self.softmax.backward(labels)
        linear_grad = self.linear.backward(softmax_grad)
        flatten_grad = self.flatten.backward(linear_grad)
        max_pool3_grad = self.max_pool3.backward(flatten_grad)
        relu3_grad = self.relu3.backward(max_pool3_grad)
        conv3_grad = self.conv3.backward(relu3_grad)
        max_pool2_grad = self.max_pool2.backward(conv3_grad)
        relu2_grad = self.relu2.backward(max_pool2_grad)
        conv2_grad = self.conv2.backward(relu2_grad)
        max_pool1_grad = self.max_pool1.backward(conv2_grad)
        relu1_grad = self.relu1.backward(max_pool1_grad)
        conv1_grad = self.conv1.backward(relu1_grad)


class FullyConnected(object):

    def __init__(self):
        self.linear1 = nn.Linear(in_dimension=3072, out_dimension=600)
        self.relu1 = nn.Relu()
        self.dropout1 = nn.Dropout(rate=0.5)
        self.linear2 = nn.Linear(in_dimension=600, out_dimension=200)
        self.relu2 = nn.Relu()
        self.dropout2 = nn.Dropout(rate=0.5)
        self.linear3 = nn.Linear(in_dimension=200, out_dimension=10)
        self.softmax = nn.Softmax()
        self.layers = [self.linear1, self.relu1, self.dropout1,
                       self.linear2, self.relu2, self.dropout2,
                       self.linear3, self.softmax]

    def __call__(self, x):
        x = self.dropout1(self.relu1(self.linear1(x)))
        x = self.dropout2(self.relu2(self.linear2(x)))
        x = self.softmax(self.linear3(x))

        return x

    def backward(self, labels):
        softmax_grad = self.softmax.backward(labels)
        linear3_grad = self.linear3.backward(softmax_grad)
        dropout2_grad = self.dropout2.backward(linear3_grad)
        relu2_grad = self.relu2.backward(dropout2_grad)
        linear2_grad = self.linear2.backward(relu2_grad)
        dropout1_grad = self.dropout1.backward(linear2_grad)
        relu1_grad = self.relu1.backward(dropout1_grad)
        linear1_grad = self.linear1.backward(relu1_grad)

