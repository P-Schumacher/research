import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
plt.style.use(['seaborn', 'thesis'])

N = 1000
M = 10
fig, ax = plt.subplots(1,1)
for typ in ['unif', 'large', 'small']:
    ret = np.zeros([M, N])
    for rep in range(M):
        error = f'errors_{typ}_{rep}.npy'
        errors = np.load(f'./buffer_data/{error}')
        intra = []
        for i in range(N):
            tmp = []
            for j in range(errors.shape[1]):
                tmp.append(np.square(errors[i,j] - np.mean(errors[i,:])))
            length = len(tmp)
            intra.append(np.sum(tmp)/length)
        ret[rep,:] = intra
    ret_mean = np.mean(ret, axis=0)
    ret_std = np.std(ret, axis=0)
    #for i in range(0, N, 100):
    #    r_mean.append(ret_mean[i])
    #    r_std.append(ret_std[i])
    x_rang = np.arange(N)
    ax.plot(x_rang, ret_mean)
    ax.fill_between(x_rang, np.maximum(0., ret_mean - ret_std), ret_mean + ret_std, alpha=0.6)
    ax.set_xlim([-0.5, 1000.5])

ax.legend(['uniform sampling', 'PER'], frameon=True)
ax.set_xlabel('training iteration')
ax.set_ylabel('TD-error variance')



        
plt.tight_layout()

plt.savefig('td_error_variability.pdf')
plt.show()
