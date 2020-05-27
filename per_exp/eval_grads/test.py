import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('GTK3Agg')
plt.style.use('seaborn')
a = np.load('lowtd.npy')
#b = np.reshape(b, [b.shape[-1],])
b = np.load('uniform.npy')
c = np.load('hightd.npy')
batch_range = np.concatenate([np.array([1, 5, 128, 256, 512]), np.arange(1000, 6000, 1000)], axis=0)
plt.plot(batch_range, a)
plt.plot(batch_range, b)
plt.plot(batch_range, c)
plt.xlim([0, 1500])
plt.legend(['lowtd','uniform', 'hightd'])
#plt.ylim([0, 1.3])
plt.show()
