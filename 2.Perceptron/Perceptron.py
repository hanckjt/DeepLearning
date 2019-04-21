import numpy as np
import matplotlib.pyplot as plt
import sys, os
from PIL import Image

sys.path.append(os.path.join(os.pardir, 'BookSource'))
from dataset.mnist import load_mnist


def step(x):
	return (x > 0).astype(np.int)


def PERC(x, w, b):
	return step(np.sum(x * w) + b)


def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.2
	return PERC(x, w, b)


def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	return PERC(x, w, b)


def NAND(x1, x2):
	return 1 - AND(x1, x2)


# multi-layered perceptron
def XOR(x1, x2):
	return AND(NAND(x1, x2), OR(x1, x2))


def get_e(x = 1000, s = False):
	if s:
		e = np.e
	else:
		e = (1 + 1 / x) ** x

	return e


def exp(x):
	e = get_e(s = True)
	return e ** x


def sigmoid(x):
	return 1 / (1 + exp(-x))


def relu(x):
	return np.maximum(0, x)


def identity(x):
	return x


def softmax(a):
	c = np.max(a)
	exp_a = exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	print(np.sum(y))
	return y


def init_network():
	network = {}


def forward(network, x):
	pass


def show_functions_plt():
	funs_list = [step, relu, sigmoid, softmax]

	plt.ylim(top = 1)

	x = np.arange(-10, 10, 0.1)

	for fun in funs_list:
		y = fun(x)
		plt.plot(x, y, label = fun.__name__)

	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.xticks(np.arange(-10, 12, 2))
	plt.axvline(0, color = 'red', linewidth = 0.5)
	plt.axhline(0.5, color = 'red', linewidth = 0.5)
	plt.plot(0, 0.5, 'ro')

	plt.title('Mix')
	plt.grid(True)
	plt.legend()
	plt.show()


def show_img(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()


if __name__ == '__main__':
	(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)
	img, label = x_train[0], t_train[0]
	img = img.reshape(28, 28)
	show_img(img)
