

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
N_true = 1000000
def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    agent._replay_buffer.load_data('./per_exp/eval_grads/buffer_data/')

    buff = agent._replay_buffer
    buff.size = N 
    buff.ptr = N
    buff.max_size = N
    buff.tree.n_entries = N
    
    #agent.load_model('./per_exp/eval_grads/model/TD3_Vrep_save_transitions')
    #max_action= tf.constant([5.585, 5.585, 3.071, 3.071, 1.919, 0.698, 0.698, 1.], dtype=tf.float32)
    #new_actor = Actor(22, 8, max_action, [300, 300], 9e-6)
    new_actor = agent._policy.actor
    agent = agent._policy
   
    for t in range(100000):
        state, action, reward, next_state, done, *_ = buff.sample_uniformly(128)
        error, *_ = agent.train(buff, 128, t, True, None)
        #print(f'td error {tf.norm(error)} iteration {i}')
        if not t % 5000:
            print(f'{t} of 100000')
            agent.actor.save_weights('converged_actor')
            wandb.log({'actor_loss':error})

