import numpy as np


def ACT(x):
	return (1 if x > 0 else 0)


def PERC(x, w, b):
	return ACT(np.sum(x * w) + b)


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


print(XOR(0, 1))
