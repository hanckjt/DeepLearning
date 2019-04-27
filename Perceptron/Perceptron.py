import numpy as np
import matplotlib.pyplot as plt


class DrawFuctionPlot:
	funsList = []
	xMin = -6
	xMax = 6
	xStep = 0.1
	yLim = 1

	def __init__(self, yLim = 1):
		self.yLim = yLim

	def setXAxis(self, xMin, xMax, xStep):
		self.xMin, self.xMax, self.xStep = xMin, xMax, xStep

	def addFunction(self, fun):
		self.funsList.append(fun)

	def clearFunctions(self):
		self.funsList.clear()

	def draw(self):
		plt.ylim(top = self.yLim)

		x = np.arange(self.xMin, self.xMax + self.xStep, self.xStep)

		for fun in self.funsList:
			y = fun(x)
			plt.plot(x, y, label = fun.__name__)

		plt.axvline(0, color = 'red', linewidth = 0.5)
		plt.axhline(0.5, color = 'red', linewidth = 0.5)
		plt.plot(0, 0.5, 'ro')

		plt.grid(True)
		plt.legend()
		plt.show()


def step(x):
	return (x > 0).astype(np.int)


def PROC(x, w, b):
	return step(np.sum(x * w) + b)


def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.2
	return PROC(x, w, b)


def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	return PROC(x, w, b)


def NAND(x1, x2):
	return 1 - AND(x1, x2)


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
	return 1 / (1 + np.exp(-x))



def sigmoid_grad(x):
	return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
	return np.maximum(0, x)


def identity(x):
	return x



def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis = 0)
		y = np.exp(x) / np.sum(np.exp(x), axis = 0)
		return y.T

	x = x - np.max(x)  # 溢出对策
	return np.exp(x) / np.sum(np.exp(x))


def log(x):
	return np.log(x)


if __name__ == '__main__':
	dp = DrawFuctionPlot()

	dp.addFunction(sigmoid)
	dp.addFunction(sigmoid_grad)
	dp.addFunction(softmax)

	dp.draw()
