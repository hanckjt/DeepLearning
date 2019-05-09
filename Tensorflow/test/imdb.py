import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
