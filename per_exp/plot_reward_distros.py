from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('GTK3Agg')
plt.style.use(['seaborn', 'thesis'])
fig, ax = plt.subplots(1,1, figsize=(6, 5))
#m1 = np.load('./buffer_data/m1.npy')
#m2 = np.load('./buffer_data/m2.npy') 
rew = np.load('./buffer_data/reward.npy')
ax.plot(np.arange(0,1000), rew[:1000])
new_rew = []
for re in rew[:1000]:
    if re == -1.:
        new_rew.append(re)
    else:
        for i in range(10):
            new_rew.append(re)
ax.plot(new_rew)
ax.set_xlabel('transition number')
ax.set_ylabel('high-level reward')
ax.set_xlim([0,1000])
ax.legend(['original distribution', 'broadened distribution'], frameon=True)
plt.savefig('reward_and_multipledones.pdf')
plt.show()
