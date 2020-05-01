import numpy as np
import tensorflow as tf
import wandb
from pudb import set_trace
from scipy import stats

from utils.utils import create_world
from matplotlib import pyplot as plt

N = 1000
def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    agent.meta_replay_buffer.load_data('./per_exp/buffer_data/')
    buff = agent.meta_replay_buffer
    buff.size = N 
    buff.max_size = N
    buff.tree.n_entries = N

    idx = np.where(buff.reward == -1.)[0]
    buff.reward[idx] = -1.

    m1 = []
    m2 = []


    idxs = np.zeros([N, N])
    for t in range(N):
        print(f'train {t} of 10000')
        agent._meta_agent.train(buff, 128, t, False, None)
        print(buff.batch_idxs)
        for i in range(buff.batch_idxs.shape[0]):
                m1.append(buff.batch_idxs[i])
                m2.append(t)
    np.save('m1.npy', m1)    
    np.save('m2.npy', m2)    
    env.close()

