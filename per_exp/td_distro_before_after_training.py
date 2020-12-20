import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
plt.style.use(['seaborn', 'thesis'])
plt.figure(figsize=(6, 5)),

errors = np.load('./buffer_data/errors.npy')
rew = np.load('./buffer_data/reward.npy')
#vals = []
#baseline = np.mean(errors[0, :])
#for i in range(1, errors.shape[0]):
#    vals.append(np.mean(errors[i, :]) - baseline)
plt.plot(errors[0,:])
plt.plot(errors[-1,:])
plt.xlabel('transition $j$')
plt.ylabel('TD-error $\delta_{j}$')
plt.xlim([0,1000])
plt.legend(['t = 0', 't = 1000'], loc='upper right', frameon=True)
plt.tight_layout()

plt.savefig('unif_td_error.pdf')
plt.show()
