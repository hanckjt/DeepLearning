import numpy as np
import matplotlib.pyplot as plt
import sys, os
from PIL import Image

sys.path.append(os.path.join(os.pardir))
sys.path.append(os.path.join(os.pardir, 'BookSource'))
from dataset.mnist import load_mnist
from 2.Perceptron import Perceptron

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    d = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + d)) / batch_size


def numerical_diff(f, x):
    h = 1e-7
    return (f(x + h) - f(x - h)) / (2 * h)


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]