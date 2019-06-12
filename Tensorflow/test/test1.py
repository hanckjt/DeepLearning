import sys, os
import wx
import tensorflow as tf
from tensorflow import keras

sys.path.append('D:/Workdir/DeepLearning')
from MnistTest import MnistTest


def trainModel(x_train, y_train):
    layers = [
        keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation=keras.activations.relu),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.5),

        keras.layers.Conv2D(128, (3, 3), input_shape=(28, 28, 1), activation=keras.activations.relu),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.5),

        keras.layers.Conv2D(256, (3, 3), input_shape=(28, 28, 1), activation=keras.activations.relu),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(10, activation=keras.activations.softmax)
    ]

    model = keras.models.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=1000)

    return model


class TestNet:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return model.predict(data.reshape((1, 28, 28, 1)))


if __name__ == '__main__':
    model_file = 'mnist.mod'
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0, x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0
        model = trainModel(x_train, y_train)
        model.evaluate(x_test, y_test)
        model.save(model_file)

    print(model.summary())

    app = wx.App()
    frame = MnistTest.MnistTestWindow(TestNet(model))
    frame.Show()
    app.MainLoop()
