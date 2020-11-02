import numpy as np
from pudb import set_trace
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rc
plt.style.use(['seaborn', 'thesis'])

for string in ['unif','large', 'small']:
    fig, ax = plt.subplots(figsize=(5,5))
    errors = np.load(f'./buffer_data/errors_{string}.npy')

    ax.plot(errors[0,:])
    ax.plot(errors[-1,:])
    ax.set_xlabel('i-th transition')
    ax.set_ylabel('i-th TD-error')
    ax.set_xlim([0,1000])
    ax.set_ylim([0, 30])
    plt.legend(['t = 0', 't = 1000'], loc='upper right', frameon=True)

    plt.savefig(f'{string}_td_error.pdf')
plt.show()
