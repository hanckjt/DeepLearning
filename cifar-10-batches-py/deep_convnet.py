# coding: utf-8
import sys, os
import pickle
import numpy as np
from common.layers import *


class ConvParams:
    def __init__(self, filter_num=8, filter_size=3, pad=1, stride=1):
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride

    def __str__(self):
        return 'filter_num:%d, filter_size:%d, pad:%d, stride:%d' % (self.filter_num, self.filter_size, self.pad, self.stride)


class MultiDeepConvNet:
    # convs_param: class ConvParams list
    # net_layers: net layers struct dict,
    # keys include 'conv','relu', 'pool',
    def __init__(self, input_dim, convs_param, hidden_size=50, output_size=10):
        pre_channel_num = input_dim[0]
        pre_node_num = pre_channel_num * convs_param[0].filter_size ** 2
        self.params = []
        for conv in convs_param:
            weight_init_scale = np.sqrt(2.0 / pre_node_num)
            W = weight_init_scale * np.random.randn(conv.filter_num, pre_channel_num, conv.filter_size, conv.filter_size)
            b = np.zeros(conv.filter_num)
            self.params.append(W)
            self.params.append(b)
            pre_channel_num = conv.filter_num
            pre_node_num = pre_channel_num * conv.filter_size ** 2
        # hidden layer
        self.params.append(np.sqrt(2.0 / pre_node_num) * np.random.randn(convs_param[-1].filter_num * (convs_param[-1].filter_size + convs_param[-1].pad) ** 2, hidden_size))
        self.params.append(np.zeros(hidden_size))
        # out layer
        pre_node_num = hidden_size
        self.params.append(np.sqrt(2.0 / pre_node_num) * np.random.randn(hidden_size, output_size))
        self.params.append(np.zeros(output_size))

        # make layers
        self.layers = []
        for idx in range(0, len(convs_param), 1):
            W = self.params[idx * 2]
            b = self.params[idx * 2 + 1]
            stride = convs_param[idx].stride
            pad = convs_param[idx].pad
            self.layers.append(Convolution(W, b, stride, pad))
            self.layers.append(Relu())
            if idx % 2 == 1:
                self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params[-4], self.params[-3]))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params[-2], self.params[-1]))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Convolution) or isinstance(layer, Affine):
                grads['W' + str(i + 1)] = layer.dW
                grads['b' + str(i + 1)] = layer.db

        return grads

    def save_params(self, file_name="convnet_params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="convnet_params.pkl"):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

        idx = 1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Convolution) or isinstance(layer, Affine):
                self.layers[i].W = self.params['W' + str(idx)]
                self.layers[i].b = self.params['b' + str(idx)]
                idx = idx + 1
