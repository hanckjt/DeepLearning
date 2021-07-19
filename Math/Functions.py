import numpy as np

from DrawFunction import *


def linear_fn(x):
    w = 1
    b = 0
    y = w * x + b

    return y


def quadratic_fn(x):
    y = x ** 2

    return y


def sigmod_fn(x):
    y = 1 / (1 + np.exp(-x))

    return y


def normal_dist_fn(x):
    a = 1
    b = 0
    y = 1 / (np.sqrt(2 * np.pi) * a) * np.exp(-((x - b) ** 2 / 2 * (a ** 2)))

    return y


def sin_fn(x):
    y = np.sin(x)

    return y


def cos_fn(x):
    y = np.cos(x)

    return y

def x2_fn(x):
    y = (quadratic_fn(x) - quadratic_fn(x+0.00001))/0.00001 - 1

    return y


if __name__ == '__main__':
    dfp = DrawFuctionPlot()

    dfp.addFunction(linear_fn)
    dfp.addFunction(quadratic_fn)
    dfp.addFunction(x2_fn)
    dfp.addFunction(sigmod_fn)
    dfp.addFunction(normal_dist_fn)
    dfp.addFunction(sin_fn)
    dfp.addFunction(cos_fn)

    dfp.draw()
