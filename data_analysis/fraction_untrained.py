import numpy as np
from matplotlib import pyplot as plt
M = 100
size = 100000

N_untr = 30000

results = []
for size in range(30000, 100000, 10000):
    tmp = 0
    for i in range(M):
        buff = np.zeros(shape=[size,])
        buff[:N_untr] = 1
        tmp += np.sum(np.sum(np.random.choice(buff, size=128)))
    results.append(tmp/(M*128))
plt.plot(np.linspace(50000, 200000, len(results)), results)
plt.xlabel('Total transitions in buffer')
plt.ylabel('Fraction of untrained transitions in batch')
plt.show()

'''Plots average number of states where sub-agent is untrained in one batch of 128 transitions, if we assume that the
sub-agent needs 300000 steps to be trained and we have 30000 untrained transitions in the meta-agent replay buffer
(c=10), for different numbers of total transitions. I.e. after 1e6 environment steps, the number of untrained
transitions is way down.''' 
