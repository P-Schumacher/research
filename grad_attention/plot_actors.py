import numpy as np
from matplotlib import pyplot as plt
from pudb import set_trace
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['seaborn','thesis_small'])
mpl.rcParams['axes.grid'] = 0

def standardize(A):
    return (A - np.mean(A)) / np.std(A)

class Accum:
    def __init__(self):
        self.reset()

    def collect(self, grad):
        if not self.init:
            self.grad_collect = grad
            self.init = True
        else:
            self.grad_collect = np.concatenate([self.grad_collect, grad], 0)

    def get_output(self, pos=False):
        if pos:
            return np.abs(self.grad_collect)
        else:
            return self.grad_collect

    def reset(self):
        self.init = False


max_N = 500
files = ['c5_it','c10_it', 'c10_it']
cmap = plt.cm.viridis

# Compute shared max-min for colorbar 
for level in ['meta']:
    for component in ['critic']:
        ims = []
        for idx, folder in enumerate(files):
            acc = Accum()
            for i in range(0, max_N, 1):
                a = np.load(f'{folder}/grad_{component}_{level}_{i}.npy')
                #a = standardize(a)
                if not np.any(np.isnan(a)):
                    a = np.reshape(a, [1,a.shape[0]])
                    a = a[:,:26]
                    acc.collect(a)
            im = acc.get_output()
            ims.append(im)
        combined_data = np.array(ims)
        #Get the min and max of all your data
        _min, _max = np.amin(combined_data), np.amax(combined_data)
        globals()[f'{level}_{component}_min'] = _min
        globals()[f'{level}_{component}_max'] = _max


gridspec = {'width_ratios': [1, 0.1]}
for idx, folder in enumerate(files):
    fig, ax = plt.subplots(1, 2, figsize=(10, 20), gridspec_kw=gridspec)
    acc = Accum()
    for i in range(0, max_N, 1):
        a = np.load(f'{folder}/grad_critic_meta_{i}.npy')
        #a = standardize(a)
        if not np.any(np.isnan(a)):
            a = np.reshape(a, [1,a.shape[0]])
            a = a[:, :26]
            acc.collect(a)
    im = acc.get_output([1 if idx == 2 else 0][0])
    print(im.shape)
    #c_im = ax[idx].imshow(im, cmap=cmap, vmin=meta_critic_min, vmax=meta_critic_max)
    c_im = ax[0].imshow(im, cmap=cmap)
    labels = ['EEx', 'EEy', 'EEz', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    labels = labels + [f'vel_{x}' for x in labels]
    labels = labels + ['b1x', 'b1y', 'b1F', 'b2x', 'b2y', 'b2F']#, 'A0','A1', 'A2']
    ax[0].set_xticks(np.arange(26))
    ax[0].set_xticklabels(labels, rotation=45)
    ax[0].set_ylabel('high-level training iteration')
    fig.colorbar(c_im, cax=ax[-1])
    ax[0].set_aspect('auto')
    plt.tight_layout()
    plt.savefig(f'attention_pretrained_{idx}.pdf')
    plt.show()
