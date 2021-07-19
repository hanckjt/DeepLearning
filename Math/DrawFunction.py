import numpy as np
import matplotlib.pyplot as plt


class DrawFuctionPlot:
    funsList = []
    xMin = -6
    xMax = 6
    xStep = 0.1
    yLim = 1

    def __init__(self, yLim=1):
        self.yLim = yLim

    def setXAxis(self, xMin, xMax, xStep):
        self.xMin, self.xMax, self.xStep = xMin, xMax, xStep

    def addFunction(self, fun):
        self.funsList.append(fun)

    def clearFunctions(self):
        self.funsList.clear()

    def draw(self):
        plt.ylim(top=self.yLim, bottom=-self.yLim)

        x = np.arange(self.xMin, self.xMax + self.xStep, self.xStep)

        for fun in self.funsList:
            y = fun(x)
            plt.plot(x, y, label=fun.__name__)

        plt.axvline(0, color='red', linewidth=0.5)
        plt.axhline(0, color='red', linewidth=0.5)
        plt.plot(0, 0, 'ro')

        plt.grid(True)
        plt.legend()
        plt.show()
