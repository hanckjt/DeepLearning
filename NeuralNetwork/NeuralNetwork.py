import numpy as np

import matplotlib.pyplot as plt
import sys, os
from PIL import Image
import wx
import pickle

sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir, 'BookSource'))
from dataset.mnist import load_mnist
from MnistTest import MnistTest


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    d = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + d)) / batch_size


def tangent(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y


def _numerical_diff_num(f, x):
    h = 1e-7
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_diff_array(f, x):
    h = 1e-7
    return np.array([(f(t + h) - f(t - h)) / (2 * h) for t in x])


def numerical_diff(f, x):
    if isinstance(x, float) or isinstance(x, int):
        return _numerical_diff_num(f, x)

    return _numerical_diff_array(f, x)


def numerical_gradient(f, x):
    grad = np.array([numerical_diff(f, t) for t in x])
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        print('x:', x)
        x -= lr * grad

    return x


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))

        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class MultiLayerNet:
    def __init__(self, neurals_size, weight_init_std=0.01):
        self.params = []
        self.layers = []

        for i in range(0, len(neurals_size) - 1):
            W = weight_init_std * np.random.randn(neurals_size[i], neurals_size[i + 1])
            b = np.zeros(neurals_size[i + 1])
            self.params.append(W)
            self.params.append(b)
            self.layers.append(Affine(W, b))
            if i < len(neurals_size) - 2:
                self.layers.append(Relu())

        self.lastLayer = SoftmaxWithLoss()  # output

    def predict(self, x):
        float_array = False
        if x.ndim == 1:
            x = x.reshape(-1, x.shape[0])
            float_array = True

        for layer in self.layers:
            x = layer.forward(x)

        if float_array:
            x = softmax(np.reshape(x, x.shape[1]))

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        a = np.sum(y == t) / float(x.shape[0])

        return a

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = []
        for p in self.params:
            grads.append(numerical_gradient(loss_W, p))

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        grads = []
        re_layers = self.layers.copy()
        re_layers.reverse()

        for layer in re_layers:
            dout = layer.backward(dout)
            if isinstance(layer, Affine):
                grads.append(layer.db)
                grads.append(layer.dW)

        grads.reverse()

        return grads

    def train(self, trains, tests, iters_num, batch_size, call_back=None, learning_rate=0.1, print_log=False):
        x_train, t_train = trains[0], trains[1]
        x_test, t_test = tests[0], tests[1]

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grad = self.gradient(x_batch, t_batch)

            for idx in range(0, len(self.params)):
                self.params[idx] -= learning_rate * grad[idx]

            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            train_acc = 0
            test_acc = 0
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                if print_log:
                    print("Process:{:.0f}%, TrainAcc:{:.2f}%,, TestAcc:{:.2f}%".format((i / iters_num) * 100, train_acc * 100, test_acc * 100))

            if call_back is not None:
                call_back(i, train_acc, test_acc)

        return train_loss_list, train_acc_list, test_acc_list


def trainMnistNetwork():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    print('Create a Mnist Network')
    net = MultiLayerNet(neurals_size=[784, 50, 10])
    print('Begin Training Mnist Network!')
    net.train(trains=(x_train, t_train), tests=(x_test, t_test), iters_num=10000, batch_size=100, learning_rate=0.1, print_log=True)
    print('Training Done!')

    return net


def getNetwork(fileName):
    try:
        netPickle = open(fileName, 'rb')
        net = pickle.load(netPickle)
    except FileNotFoundError:
        net = trainMnistNetwork()
        netPickle = open(fileName, 'wb')
        pickle.dump(net, netPickle)

    netPickle.close()
    return net


if __name__ is '__main__':
    app = wx.App()
    frame = MnistTest.MnistTestWindow(getNetwork('mnist_network.nur'))
    frame.Show()
    app.MainLoop()
