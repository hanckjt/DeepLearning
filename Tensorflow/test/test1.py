import sys, os
import wx
import tensorflow as tf
from tensorflow import keras

sys.path.append('D:/Workdir/DeepLearning')
from MnistTest import MnistTest


def trainModel(x_train, y_train):
    layers = [keras.layers.Conv2D(16, (3, 3), input_shape=(28, 28), activation=keras.activations.relu),
              keras.layers.Conv2D(16, (3, 3), input_shape=(28, 28), activation=keras.activations.relu),
              keras.layers.MaxPool2D(),
              keras.layers.Flatten(),
              keras.layers.Dense(512, activation=keras.activations.relu),
              keras.layers.Dropout(0.2),
              keras.layers.Dense(10, activation=keras.activations.softmax)]

    model = keras.models.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train.reshape((1, x_train.shape[0], 28, 28)), y_train.reshape((1, y_train[0], 1)), epochs=5)

    return model


class TestNet:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return model.predict(data.reshape((1, 28, 28)))

if __name__ == '__main__':
    model_file = 'mnist.mod'
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = trainModel(x_train, y_train)
        model.evaluate(x_test, y_test)
        model.save(model_file)

    print(model.summary())

    app = wx.App()
    frame = MnistTest.MnistTestWindow(TestNet(model))
    frame.Show()
    app.MainLoop()
