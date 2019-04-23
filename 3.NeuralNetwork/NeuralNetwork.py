import numpy as np
import matplotlib.pyplot as plt
import sys, os
from PIL import Image

sys.path.append(os.path.join(os.pardir, 'BookSource'))
from dataset.mnist import load_mnist