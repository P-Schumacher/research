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
a = np.load('lowtd_crit50_10000_samp10.npy')
print(a)
#b = np.reshape(b, [b.shape[-1],])
c = np.load('uniform_crit50_10000_samp10.npy')
batch_range = np.array([1, 64, 128, 256, 1000, 2000])
plt.plot(batch_range, a)
#plt.plot(batch_range, b)
plt.plot(batch_range, c)
plt.legend(['Small TD Prio.', 'Uniform Prio.'])
plt.ylim([-0.1, 1.])
plt.xlabel('Batch Size')
plt.ylabel('Avg. Cos. Sim. with true grad')
plt.xticks(batch_range, batch_range)
plt.xlim([0, 1000])
plt.title('Critic training iterations: 50')
plt.savefig('critic50.pdf')
plt.show()
