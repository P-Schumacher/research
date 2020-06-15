import numpy as np
from matplotlib import pyplot as plt

y = np.load('scatter_metric.npy')
x = np.load('scatter_td_error.npy')

plt.scatter(x, y)
plt.show()

