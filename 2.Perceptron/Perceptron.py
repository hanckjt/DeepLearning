import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
	x = np.arange(-6, 6, 0.1)
	y_sigmoid = sigmoid(x)
	y_step = step(x)

	plt.plot(x, y_sigmoid, label = 'Sigmoid')
	plt.plot(x, y_step, '--', label = 'Step')

	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.xticks(np.arange(-6, 8, 2))
	plt.axvline(0, color = 'red', linewidth = 0.5, linestyle = '--')
	plt.axhline(0.5, color = 'red', linewidth = 0.5, linestyle = '--')
	plt.plot(0, 0.5, 'ro')

	plt.title('Sigmoid')
	plt.grid(True)
	plt.legend()
	plt.show()
