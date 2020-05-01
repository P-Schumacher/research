import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
#plt.style.use('seaborn')

plt.subplot(121)
m1 = np.load('./buffer_data/m1.npy')
m2 = np.load('./buffer_data/m2.npy')

rew = np.load('./buffer_data/reward.npy')


print(m1.shape)
print(m2.shape)


xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
plt.plot(m1, m2, 'k.', markersize=2)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('Transition')
plt.ylabel('High-level training iterations')


plt.subplot(122)
fig, ax = plt.subplots()

ax.grid()
ax.set_facecolor('.8')
ax.tick_params(length=0)
ax.grid(True, axis='x', color='white')
ax.set_axisbelow(True)
[spine.set_visible(False) for spine in ax.spines.values()]
plt.plot(rew[:1000])
plt.show()
