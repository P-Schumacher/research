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
matplotlib.use('GTK3Agg')
plt.style.use('seaborn')

batch_range = [1, 32, 64, 128, 256, 512, 1024, 2048]
set_trace()
per = np.load('simils_high.npy')
per = np.reshape(per, [1, per.shape[0]])
plot_boot(batch_range, per, 'r')
per = np.load('simils_unif.npy')
per = np.reshape(per, [1, per.shape[0]])
plot_boot(batch_range, per, 'b')
per = np.load('simils_low.npy')
per = np.reshape(per, [1, per.shape[0]])
plot_boot(batch_range, per, 'k')

plt.xlabel('Batch Size')
plt.ylabel('Avg. Cos. Sim. with Hlgh-Quality Grad.')
plt.xticks(batch_range, batch_range)
#plt.xlim([0, 1000])
plt.title('True critic: 1000 Critic: 5')
plt.legend(['high td', 'unif', 'low'])
plt.show()
