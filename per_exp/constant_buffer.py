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
    print(buff.alpha)
    new_rew = []
    if True:
        co = 0
        for re in buff.reward[:N]:
            if re == -1.:
                new_rew.append(re)
            else:
                for i in range(1):
                    new_rew.append(re)
        buff.reward = np.asarray(new_rew)

    buff.max_size = N
    buff.tree.n_entries = N

    idx = np.where(buff.reward == -1.)[0]

    m1 = []
    m2 = []

    counter = 0
    idxs = np.zeros([N, N])

    errors = np.zeros([N, N])
    for t in range(N):
        #print(f'train {t} of 10000')
        agent._meta_agent.train(buff, 5, t, False, None)
        #print(buff.batch_idxs)
        state, action, reward, next_state, done = buff.get_buffer() 
        error = agent._meta_agent._compute_td_error(state, action, reward, next_state, done)
        errors[t] = error[:, 0]



        for i in range(buff.batch_idxs.shape[0]):
                if buff.reward[buff.batch_idxs[i]] != -1.:
                    counter += 1
                    print(counter)
                print(f' counter {counter}')
                m1.append(buff.batch_idxs[i])
                m2.append(t)
    np.save('m1.npy', m1)    
    np.save('m2.npy', m2)    
    np.save('errors.npy', errors)
    env.close()

