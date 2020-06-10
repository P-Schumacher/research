import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
matplotlib.use('GTK3Agg')
plt.style.use('seaborn')
#plt.figure(figsize=(10, 4)),
plt.figure(figsize=(5, 5)),

#plt.subplot(121)
m1 = np.load('./buffer_data/m1.npy')
m2 = np.load('./buffer_data/m2.npy')
errors = np.load('./buffer_data/errors.npy')
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
kernel = stats.gaussian_kde(values, bw_method=0.05)
Z = np.reshape(kernel(positions).T, X.shape)

#plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#          extent=[xmin, xmax, ymin, ymax])
#plt.plot(m1, m2, 'k.', markersize=1, marker='o')
#cb = plt.colorbar()
#cb.set_ticks([np.min(Z), (np.max(Z)+np.min(Z))/2, np.max(Z)])
#cb.set_ticklabels([0, 0.5, 1])
#plt.xlim([xmin, xmax+1])
#plt.ylim([ymin, ymax])
#plt.xlabel('Transition')
#plt.ylabel('High-level training iterations')
#plt.title('Transition density map')
#plt.subplot(122)
#plt.plot(rew[:1000])
new_rew = []
for re in rew[:1000]:
    if re == -1.:
        new_rew.append(re)
    else:
        for i in range(1):
            new_rew.append(re)
plt.plot(errors[0,:])
plt.plot(errors[-1,:])
#plt.plot(new_rew)
plt.xlabel('Transition')
#plt.ylabel('High-level reward')
plt.ylabel('TD-error $\delta_{k}$')
plt.xlim([0,1000])
plt.legend(['t = 0', 't = 1000'], loc='upper right')

#plt.legend(['Original distribution', 'Broadened distribution'])
#plt.title('Reward distribution in the buffer')
plt.title('TD-error distribution before and after training')
plt.savefig('uniform_td_error.pdf')
plt.show()
