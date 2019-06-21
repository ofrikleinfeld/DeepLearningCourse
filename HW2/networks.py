import modules as nn


class CNN(object):

    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.sigmoid1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(rate=0.3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=3)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=2, stride=2)
        self.sigmoid3 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(rate=0.4)
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=2)
        self.sigmoid4 = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_dimension=80, out_dimension=40)
        self.sigmoid5 = nn.Sigmoid()
        self.dropout3 = nn.Dropout(rate=0.5)
        self.linear_softmax = nn.LinearWithSoftmax(in_dimension=40, out_dimension=10)
        self.layers = [self.conv1, self.sigmoid1, self.dropout1,
                       self.conv2, self.sigmoid2,
                       self.conv3, self.sigmoid3, self.dropout2,
                       self.conv2, self.sigmoid4, self.flatten,
                       self.linear1, self.sigmoid5, self.dropout3,
                       self.linear_softmax
                       ]

    def __call__(self, x):
        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.sigmoid2(x)
        x = self.conv3(x)
        x = self.dropout2(x)
        x = self.sigmoid3(x)
        x = self.conv4(x)
        x = self.sigmoid4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.sigmoid5(x)
        x = self.dropout3(x)
        x = self.linear_softmax(x)

        return x

    def backward(self, labels):
        layer_L_grad = self.linear_softmax.backward(labels)
        linear1_grad = self.linear1.backward(self.linear_softmax.weights, layer_L_grad) * self.sigmoid5.backward()
        flatten_grad = self.flatten.backward(self.linear1.weights, linear1_grad) * self.sigmoid4.backward()
        conv4_grad = self.conv4.backward(flatten_grad) * self.sigmoid3.backward()
        conv3_grad = self.conv3.backward(conv4_grad) * self.sigmoid2.backward()
        conv2_grad = self.conv2.backward(conv3_grad) * self.sigmoid1.backward()
        conv1_grad = self.conv1.backward(conv2_grad)


class SmallCNN(object):

    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1)
        self.relu1 = nn.Relu()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1)
        self.relu2 = nn.Relu()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_dimension=1728, out_dimension=200)
        self.relu3 = nn.Relu()
        self.linear2 = nn.Linear(in_dimension=200, out_dimension=10)
        self.softmax = nn.Softmax()
        self.layers = [self.conv1, self.relu1,
                       self.conv2, self.relu2, self.flatten,
                       self.linear1, self.relu3,
                       self.linear2, self.softmax
                       ]

    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.softmax(self.linear2(x))

        return x

    def backward(self, labels):
        softmax_grad = self.softmax.backward(labels)
        linear2_grad = self.linear2.backward(softmax_grad)
        relu3_grad = self.relu3.backward(linear2_grad)
        linear1_grad = self.linear1.backward(relu3_grad)
        flatten_grad = self.flatten.backward(linear1_grad)
        relu2_grad = self.relu2.backward(flatten_grad)
        conv2_grad = self.conv2.backward(relu2_grad)
        relu1_grad = self.relu1.backward(conv2_grad)
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

