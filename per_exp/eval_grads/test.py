import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
plt.figure(figsize=(8,10))
matplotlib.use('GTK3Agg')
plt.style.use('seaborn')
unif_mean = np.load('simil_unif_mean_20_1000.npy')
unif_std = np.load('simil_unif_std_20_1000.npy')
per_mean = np.load('simil_per_mean_20_1000.npy')
per_std = np.load('simil_per_std_20_1000.npy')
batch_range = [1, 10, 128, 256, 1000]
plt.subplot(131)
plt.plot(batch_range, unif_mean, color='b')
plt.fill_between(batch_range, unif_mean - unif_std, unif_mean + unif_std,
   color='b', alpha=0.2)
plt.plot(batch_range, per_mean, color='r')
plt.fill_between(batch_range, per_mean - per_std, per_mean + per_std,
   color='r', alpha=0.2)
plt.subplot(132)
unif_mean = np.load('simil_unif_mean_1_1000.npy')
unif_std = np.load('simil_unif_std_1_1000.npy')
per_mean = np.load('simil_per_mean_1_1000.npy')
per_std = np.load('simil_per_std_1_1000.npy')
plt.plot(batch_range, unif_mean, color='b')
plt.fill_between(batch_range, unif_mean - unif_std, unif_mean + unif_std,
   color='b', alpha=0.2)
plt.plot(batch_range, per_mean, color='r')
plt.fill_between(batch_range, per_mean - per_std, per_mean + per_std,
   color='r', alpha=0.2)
plt.subplot(133)
unif_mean = np.load('simil_unif_mean_300_1000.npy')
unif_std = np.load('simil_unif_std_300_1000.npy')
per_mean = np.load('simil_per_mean_300_1000.npy')
per_std = np.load('simil_per_std_300_1000.npy')
plt.plot(batch_range, unif_mean, color='b')
plt.fill_between(batch_range, unif_mean - unif_std, unif_mean + unif_std,
   color='b', alpha=0.2)
plt.plot(batch_range, per_mean, color='r')
plt.fill_between(batch_range, per_mean - per_std, per_mean + per_std,
   color='r', alpha=0.2)
#plt.ylim([-0.3, 1.1])
plt.xlabel('Batch Size')
plt.ylabel('Avg. Cos. Sim. with Hlgh-Quality Grad.')
plt.xticks(batch_range, batch_range)
plt.xlim([0, 1000])
plt.title('True critic: 1000 Critic: 5')
#plt.savefig('critsdic5_1000_alpha05_08.pdf')
plt.show()
