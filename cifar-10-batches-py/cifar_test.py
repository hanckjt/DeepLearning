import sys, os, pickle

import numpy as np
import matplotlib.image as mp
from deep_convnet import *
from common.trainer import Trainer


def loadPickle(filename, encoding='ASCII'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f, encoding=encoding)
    except e:
        print(e)
        return None


def savePickle(filename, data):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            return True
    except e:
        print(e)
        return False


if __name__ == '__main__':
    # training data
    batch_num = 5
    x_train, t_train = np.empty(shape=(0), dtype='uint8'), np.empty(shape=(0), dtype='uint8')
    for i in range(1, batch_num + 1):
        filename = 'data_batch_%d' % i
        train = loadPickle(filename, 'bytes')
        t_train = np.append(t_train, train[b'labels'])
        x_train = np.append(x_train, train[b'data'])
    x_train = x_train.reshape((t_train.shape[0], 3, 32, 32))

    # testing data
    tests = loadPickle('test_batch', 'bytes')
    t_test = np.array(tests[b'labels'], dtype='uint8')
    x_test = np.array(tests[b'data'], dtype='uint8').reshape((t_test.shape[0], 3, 32, 32))

    # label names
    label_names = loadPickle('batches.meta')['label_names']

    convs = [ConvParams(8, 3, 1, 1),
             ConvParams(8, 3, 1, 1),
             ConvParams(16, 3, 1, 1),
             ConvParams(16, 3, 1, 1),
             ConvParams(32, 3, 1, 1),
             ConvParams(32, 3, 1, 1)]
    network = MultiDeepConvNet(input_dim=(3, 32, 32), convs_param=convs)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='rmsprpo', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
    trainer.train()
