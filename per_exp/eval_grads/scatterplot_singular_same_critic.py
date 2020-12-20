import numpy as np
import tensorflow as tf
from rl_algos.TD3_tf import Actor
import wandb
from pudb import set_trace
from scipy import stats
import copy
from utils.utils import create_world, create_agent
from matplotlib import pyplot as plt
simil_metric = tf.keras.losses.CosineSimilarity()


N = 100000 #1000000
N_TRAIN_TRUE_CRITIC = 100 #1000
SAMPLES = 5

class Accumulator:
    def __init__(self):
        self.init = False
        self.counter = 0

    def accumulate(self, grad):
        if not self.init:
            self.grad = grad
            self.counter += 1
            self.init = True
        else:
            self.grad = [x + y for x, y in zip(self.grad, grad)]
            self.counter += 1

    def get_grad(self):
        return [tf.multiply(x, 1. / self.counter) for x in self.grad]
    
    def reset(self):
        self.init = False
        self.grad = None
        self.counter = 0

def create_random_weight_list():
    weights = []
    for i in range(6):
        weights.append(tf.random.uniform([20, 300], -2, 2))
    return weights

def train_the_critic(untrained_agent, replay_buffer, iterations):
    for it in range(iterations):
        state, action, reward, next_state, done, *_ = replay_buffer.sample_uniformly(128)
        untrained_agent._policy._train_critic(state, action, reward, next_state, done, False, None)

def update_buffer(replay_buffer, agent):
    state, action, reward, next_state, done = replay_buffer.get_buffer()
    td_error = agent._policy._compute_td_error(state, action, reward, next_state, done)
    replay_buffer.batch_idxs = np.arange(0, N, 1)
    replay_buffer.tree_idxs = replay_buffer.batch_idxs + replay_buffer.tree.capacity - 1
    replay_buffer.update_priorities(td_error)

def main(cnf):
    env, agent = create_world(cnf)
    cnf_old = cnf
    cnf = cnf.main
    agent._replay_buffer.load_data('./per_exp/eval_grads/buffer_data/')
    buff = agent._replay_buffer
    buff.size = N 
    buff.ptr = N
    buff.max_size = N
    buff.tree.n_entries = N
    agent._policy.actor.load_weights('./per_exp/eval_grads/model/converged_actor')
    trained_actor = agent._policy.actor
    print('Successfully loaded')
    print('Compute true gradient of true critic')
    sim_avg = []
    td_avg = []
    for k in range(SAMPLES):
        # True gradient of true critic 
        true_critic= create_agent(cnf_old, env)
        train_the_critic(true_critic, buff, N_TRAIN_TRUE_CRITIC)
        true_critic_sample = true_critic._policy.critic
        state, *_ = buff.get_buffer()
        with tf.GradientTape() as tape:
            action = trained_actor(state)
            q_value, _  = true_critic_sample(tf.concat([state, action], axis=1))
            actor_loss = -tf.reduce_mean(q_value)
        gradients_true = tape.gradient(actor_loss, trained_actor.trainable_variables)
        gradients_true  = [tf.reshape(x, [-1]) for x in gradients_true]
        state, action, reward, next_state, done, *_ = buff.get_buffer()
        print(f' state shape {state.shape}')
        for idx, s in enumerate(state):
            print(f'{idx} of 1000000')
            s = tf.reshape(s, [1, s.shape[0]])
            ns = tf.reshape(next_state[idx], [1, next_state[idx].shape[0]])
            d = tf.reshape(done[idx], [1, done[idx].shape[0]])
            r = tf.reshape(reward[idx], [1, reward[idx].shape[0]])
            approx_critic = true_critic
            td_errors = approx_critic._policy._compute_td_error(s,  trained_actor(s), r, ns, d)
            with tf.GradientTape() as tape:
                action = trained_actor(s)
                q_value, _  = approx_critic._policy.critic(tf.concat([s, action], axis=1))
                actor_loss = -tf.reduce_mean(q_value)
            gradients_sample = tape.gradient(actor_loss, trained_actor.trainable_variables)
            gradients_sample = [tf.reshape(x, [-1]) for x in gradients_sample]
            sims = [-simil_metric(x, y) for x, y in zip(gradients_true, gradients_sample)]
            sims = tf.reduce_mean(sims)
            sim_avg.append(sims)
            td_avg.append(td_errors)
    np.save(f'sim_avg.npy', sim_avg)
    np.save(f'td_avg.npy', td_avg)




        #np.save('m1.npy', m1)    
        #np.save('m2.npy', m2)    
        #np.save('errors.npy', errors)
