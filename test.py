import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


def get_bootstrapped_values(data):
    mean_bootstrap = []
    high_bootstrap = []
    low_bootstrap = []
    for x in data:
        boot_stats = bs.bootstrap(x, stat_func=bs_stats.mean)
        mean_bootstrap.append(boot_stats.value)
        high_bootstrap.append(boot_stats.upper_bound)
        low_bootstrap.append(boot_stats.lower_bound)
    return mean_bootstrap, high_bootstrap, low_bootstrap

def plot_boot(x_range, data, color='r'):
    mean, high, low = get_bootstrapped_values(data)
    plt.plot(x_range, mean, color=color)
    plt.fill_between(x_range, low, high,
       color=color, alpha=0.2)

plt.figure(figsize=(8,10))
plt.style.use('seaborn')

batch_range = [1, 5, 64, 128, 1024]
per = np.load('simil_low.npy')
plot_boot(batch_range, per, 'r')
batch_range = [1, 5, 128, 1024]
per = np.load('simil_high.npy')
plot_boot(batch_range, per, 'b')
per = np.load('simil_uniform.npy')
plot_boot(batch_range, per, 'k')

plt.xlabel('Batch Size')
plt.ylabel('Avg. Cos. Sim. with High-Quality Grad.')
batch_range = [1, 5, 64, 128, 1024]
plt.xticks(batch_range, batch_range)
#plt.xlim([0, 1000])
plt.title('True critic: 1000 Critic: 5')
plt.legend(['low td', 'high ', 'uniform'])
plt.show()
