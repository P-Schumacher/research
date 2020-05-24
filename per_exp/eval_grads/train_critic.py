
import numpy as np
import tensorflow as tf
from rl_algos.TD3_tf import Actor
import wandb
from pudb import set_trace
from scipy import stats

from utils.utils import create_world
from matplotlib import pyplot as plt
simil_metric = tf.keras.losses.CosineSimilarity()

def create_random_weight_list():
    weights = []
    for i in range(6):
        weights.append(tf.random.uniform([20, 300], -2, 2))
    return weights

N = 1000000
def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    agent._replay_buffer.load_data('./per_exp/eval_grads/buffer_data/')
    agent._policy.actor.load_weights('./per_exp/eval_grads/model/converged_actor')
    set_trace()

    buff = agent._replay_buffer
    buff.size = N 
    buff.ptr = N
    buff.max_size = N
    buff.tree.n_entries = N
    
    agent = agent._policy
    for t in range(N+5000):
        state, action, reward, next_state, done, *_ = buff.sample_uniformly(128)
        error = agent._train_step(state, action, reward, next_state, done, False, None)
        if t == 1000:
            set_trace()
        #print(f'td error {tf.norm(error)} iteration {i}')
        if not t % 5000:
            print(f'{t} of 1005000')
            agent.critic.save_weights('converged_critic')
            #wandb.log({'tderror':tf.norm(error)})
