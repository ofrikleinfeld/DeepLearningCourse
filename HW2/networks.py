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
        self.linear2 = nn.Linear(in_dimension=40, out_dimension=10)
        self.softmax = nn.Softmax()

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
        x = self.sigmoid4(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x

    def backward(self):
        pass