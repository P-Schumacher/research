import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from sklearn import preprocessing
def normalize(Z):
    ret = Z - np.amax(Z)
    return ret / (np.amax(Z) - np.amin(Z))

plt.style.use(['seaborn', 'thesis'])
Z_collection = []
for idx, data in enumerate(['_unif', '']): 
    m1 = np.load(f'./buffer_data/m1{data}.npy')
    m2 = np.load(f'./buffer_data/m2{data}.npy') 
    rew = np.load(f'./buffer_data/reward.npy')
    print(m1.shape)
    print(m2.shape)
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    print(xmax)
    print(xmin)


    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    #Z = preprocessing.normalize(Z)
    Z = normalize(Z)
    Z_collection.append(Z)

combined_data = np.array(Z_collection)
#Get the min and max of all your data
_min, _max = np.amin(combined_data), np.amax(combined_data)

for idx, Z in enumerate(Z_collection):
    fig, ax = plt.subplots(1,1, figsize=(6, 5))
    ret = ax.imshow(np.rot90(Z), cmap=plt.cm.viridis,
              extent=[xmin, xmax, ymin, ymax], vmin=_min, vmax=_max)
    #ax.plot(m1, m2, 'k.', markersize=4, marker='o')
    ax.set_xlim([xmin, xmax+1])
    ax.set_ylim([ymin, ymax+1])
    plt.margins(x=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(ret, cax=cax)
    cb.set_ticks([np.min(Z), (np.max(Z)+np.min(Z))/2, np.max(Z)])
    cb.set_ticklabels([0, 0.5, 1])
    ax.set_xlabel('transition number')
    ax.set_ylabel('high-level training iterations')
    plt.tight_layout()
    plt.savefig(f'prio_buff_vis_{idx}.pdf')
    plt.show()

