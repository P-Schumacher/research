import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib

#matplotlib.use('GTK3Agg')
plt.style.use(['seaborn', 'thesis'])
plt.figure(figsize=(6, 5)),
m1 = np.load('./buffer_data/m1.npy')
m2 = np.load('./buffer_data/m2.npy') 
rew = np.load('./buffer_data/reward.npy')
#idx = np.where(m2 < 400)[0]
#print(idx)
#idx = np.asarray(idx, dtype=np.int32)
#print(m2[:10])
#m1 = m1[idx]
#m2 = m2[idx]
#print(m1)
#print(m2)

print(m1.shape)
print(m2.shape)


xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()
print(xmax)
print(xmin)
#plt.plot(m1, m2, 'k.', markersize=4, marker='o')
plt.hist2d(m1[:], m2[:], bins=10, cmap=plt.cm.gist_earth)
plt.savefig('prio_buff_vis.pdf')
plt.show()


#X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#positions = np.vstack([X.ravel(), Y.ravel()])
#values = np.vstack([m1, m2])
#kernel = stats.gaussian_kde(values)
#Z = np.reshape(kernel(positions).T, X.shape)
#print(np.rot90(Z).shape)
#plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth,
#          extent=[xmin, xmax, ymin, ymax])
##plt.plot(m1, m2, 'k.', markersize=4, marker='o')
#cb = plt.colorbar()
#cb.set_ticks([np.min(Z), (np.max(Z)+np.min(Z))/2, np.max(Z)])
#cb.set_ticklabels([0, 0.5, 1])
#plt.xlim([xmin, xmax+1])
#plt.ylim([ymin, ymax+1])
##plt.ylim([250, 350])
#plt.margins(x=0)
##loc, labels = plt.xticks()
##plt.xticks(ticks=loc, labels=np.arange(0, 140, 20))
##loc, labels = plt.yticks()
##plt.yticks(ticks=loc, labels=np.arange(0, 140, 20))
#plt.xlabel('transition number')
#plt.ylabel('high-level training iterations')
#plt.tight_layout()
#plt.savefig('prio_buff_vis.pdf')
#plt.show()

