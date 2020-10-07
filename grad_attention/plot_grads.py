import numpy as np
from matplotlib import pyplot as plt
from pudb import set_trace
import matplotlib as mpl
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




max_N = 100000
files = ['c10']
cmap = plt.cm.viridis


for level in ['meta', 'sub']:
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


fig, ax = plt.subplots(len(files)+1,4, figsize=(12,10))
for idx, folder in enumerate(files):
    acc = Accum()
    for i in range(0, max_N, 1000):
        a = np.load(f'{folder}/grad_critic_meta_{i}.npy')
        if not np.any(np.isnan(a)):
            a = np.reshape(a, [1,a.shape[0]])
            print(a.shape)
            acc.collect(a)
    im = acc.grad_collect
    c_im = ax[idx, 0].imshow(im, vmin=meta_critic_min, vmax=meta_critic_max)
    labels = ['EEx', 'EEy', 'EEz', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    labels = labels + [f'vel_{x}' for x in labels]
    labels = labels + ['b1x', 'b1y', 'b1F', 'b2x', 'b2y', 'b2F', 'A0','A1', 'A2']
    ax[idx, 0].set_xticks(np.arange(29))
    ax[idx, 0].set_xticklabels(labels, rotation='vertical')
    ax[idx, 0].set_ylabel('training iteration')
    ax[idx, 0].set_title(f'{folder}_critic_meta')
    #fig.colorbar(c_im, ax=ax[idx,0])

    acc.reset()
    for i in range(0, max_N, 1000):
        a = np.load(f'{folder}/grad_critic_sub_{i}.npy')
        if not np.any(np.isnan(a)):
            a = np.reshape(a, [1,a.shape[0]])
            print(a.shape)
            acc.collect(a)
    im= acc.grad_collect
    ax[idx, 1].imshow(im, vmin=sub_critic_min, vmax=sub_critic_max)
    labels = ['EEx', 'EEy', 'EEz', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    labels = labels + [f'vel_{x}' for x in labels]
    labels = labels + ['G0', 'G1', 'G2', 'A0', 'A1', 'A2', 'A3','A4', 'A5', 'A6', 'A7']
    ax[idx, 1].set_xticks(np.arange(31))
    ax[idx, 1].set_xticklabels(labels, rotation='vertical')
    ax[idx, 1].set_ylabel('training iteration')
    ax[idx, 1].set_title(f'{folder}_critic_sub')

    acc.reset()
    for i in range(0, max_N, 1000):
        a = np.load(f'{folder}/grad_actor_sub_{i}.npy')
        if not np.any(np.isnan(a)):
            a = np.reshape(a, [1,a.shape[0]])
            print(a.shape)
            acc.collect(a)
    im= acc.grad_collect
    ax[idx, 3].imshow(im, vmin=sub_critic_min, vmax=sub_critic_max)
    labels = ['EEx', 'EEy', 'EEz', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    labels = labels + [f'vel_{x}' for x in labels]
    labels = labels + ['G0', 'G1', 'G2']
    ax[idx, 3].set_xticks(np.arange(23))
    ax[idx, 3].set_xticklabels(labels, rotation='vertical')
    ax[idx, 3].set_ylabel('training iteration')
    ax[idx, 3].set_title(f'{folder}_actor_sub')

    acc.reset()
    for i in range(0, max_N, 1000):
        a = np.load(f'{folder}/grad_actor_meta_{i}.npy')
        if not np.any(np.isnan(a)):
            a = np.reshape(a, [1,a.shape[0]])
            print(a.shape)
            acc.collect(a)
    im = acc.grad_collect
    c_im = ax[idx, 2].imshow(im, vmin=meta_actor_min, vmax=meta_actor_max)
    labels = ['EEx', 'EEy', 'EEz', 'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    labels = labels + [f'vel_{x}' for x in labels]
    labels = labels + ['b1x', 'b1y', 'b1F', 'b2x', 'b2y', 'b2F']
    ax[idx, 2].set_xticks(np.arange(26))
    ax[idx, 2].set_xticklabels(labels, rotation='vertical')
    ax[idx, 2].set_ylabel('training iteration')
    ax[idx, 2].set_title(f'{folder}_actor_meta')
plt.tight_layout()
#plt.savefig('attebtion.pdf')
plt.show()
