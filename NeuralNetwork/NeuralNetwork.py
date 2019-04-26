import numpy as np
import matplotlib.pyplot as plt
import sys, os
from PIL import Image

sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir, 'BookSource'))
from dataset.mnist import load_mnist
from Perceptron.Perceptron import *


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    d = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

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


def fun_2(x):
    r = np.sum(x ** 2)
    return r


def fun_1(x):
    return x ** 2


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


if __name__ is '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    dp = DrawFuctionPlot()
    dp.setXAxis(-1, 1, 0.1)
    dp.addFunction(fun_1)
    dp.addFunction(tangent(fun_1, 0.5))
    dp.draw()

    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(fun_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    t= np.array([0, 0, 1])
    print( net.loss(x, t) )