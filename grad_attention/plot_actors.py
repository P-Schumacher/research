import numpy as np
from matplotlib import pyplot as plt
from pudb import set_trace
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['seaborn','thesis_small'])
mpl.rcParams['axes.grid'] = 0

class Accum:
    def __init__(self):
        self.reset()

    def collect(self, grad):
        if not self.init:
            self.grad_collect = grad
            self.init = True
        else:
            self.grad_collect = np.concatenate([self.grad_collect, grad], 0)

    def reset(self):
        self.init = False




max_N = 9000
#max_N = 55000 
files = ['c1', 'c10', 'c30']
files = ['c10']
cmap = plt.cm.viridis


for level in ['meta']:
    for component in ['actor', 'critic']:
        ims = []
        for idx, folder in enumerate(files):
            acc = Accum()
            for i in range(0, max_N, 1000):
                a = np.load(f'{folder}/grad_{component}_{level}_{i}.npy')
                if not np.any(np.isnan(a)):
                    a = np.reshape(a, [1,a.shape[0]])
                    acc.collect(a)
            im = acc.grad_collect
            ims.append(im)
        combined_data = np.array(ims)
        #Get the min and max of all your data
        _min, _max = np.amin(combined_data), np.amax(combined_data)
        globals()[f'{level}_{component}_min'] = _min
        globals()[f'{level}_{component}_max'] = _max


gridspec = {'width_ratios': [1, 1, 1, 0.1]}
fig, ax = plt.subplots(1, 4, figsize=(20, 20), gridspec_kw=gridspec)
for idx, folder in enumerate(files):
    acc = Accum()
    for i in range(0, max_N, 1000):
        a = np.load(f'{folder}/grad_critic_meta_{i}.npy')
        if not np.any(np.isnan(a)):
            a = np.reshape(a, [1,a.shape[0]])
            acc.collect(a)
    im = acc.grad_collect
    print(im.shape)
    c_im = ax[idx].imshow(im, cmap=cmap, vmin=meta_critic_min, vmax=meta_critic_max)
    labels = ['EEx', 'EEy', 'EEz', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    labels = labels + [f'vel_{x}' for x in labels]
    labels = labels + ['b1x', 'b1y', 'b1F', 'b2x', 'b2y', 'b2F', 'A0','A1', 'A2']
    ax[idx].set_xticks(np.arange(29))
    ax[idx].set_xticklabels(labels, rotation='vertical')
    ax[idx].set_title(f'{folder}_critic_meta')
    if idx == 0:
        ax[idx].set_ylabel('high-level training iteration')
    if idx == len(files) - 1:
        #divider = make_axes_locatable(ax[idx])
        #cax = divider.append_axes("right", size="10%", pad=0.1)
        fig.colorbar(c_im, cax=ax[-1])
    ax[idx].set_aspect('auto')

#plt.tight_layout()
#plt.savefig('attebtion.pdf')
plt.show()
