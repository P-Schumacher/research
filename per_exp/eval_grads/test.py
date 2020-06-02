import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
plt.figure(figsize=(5,5))
matplotlib.use('GTK3Agg')
plt.style.use('seaborn')
per_mean = np.load('simil_per_mean.npy')
per_std = np.load('simil_per_std.npy')
unif_mean = np.load('simil_unif_mean.npy')
unif_std = np.load('simil_unif_std.npy')
batch_range = np.array([1, 64, 128, 256, 1000])
plt.plot(batch_range, unif_mean, color='b')
plt.fill_between(batch_range, unif_mean - unif_std, unif_mean + unif_std,
   color='b', alpha=0.2)
plt.plot(batch_range, per_mean, color='r')
plt.fill_between(batch_range, per_mean - per_std, per_mean + per_std,
   color='r', alpha=0.2)
plt.legend(['Uniform Prio.', 'Small TD Prio.'])
plt.ylim([-0.1, 1.])
plt.xlabel('Batch Size')
plt.ylabel('Avg. Cos. Sim. with Hlgh-Quality Grad.')
plt.xticks(batch_range, batch_range)
plt.xlim([0, 1000])
plt.title('Critic training iterations: 10')
#plt.savefig('critic10.pdf')
plt.show()
