import numpy as np
import tensorflow as tf
from rl_algos.TD3_tf import Actor
import wandb
from pudb import set_trace
from scipy import stats
import copy
from utils.utils import create_world
from matplotlib import pyplot as plt
simil_metric = tf.keras.losses.CosineSimilarity()

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
        print(f'Critic trained {it} of {iterations}')
        state, action, reward, next_state, done, *_ = replay_buffer.sample_uniformly(128)
        untrained_agent._policy._train_critic(state, action, reward, next_state, done, False, None)

def update_buffer(replay_buffer, agent):
    state, action, reward, next_state, done = replay_buffer.get_buffer()
    td_error = agent._policy._compute_td_error(state, action, reward, next_state, done)
    replay_buffer.batch_idxs = np.arange(0, N, 1)
    replay_buffer.tree_idxs = replay_buffer.batch_idxs + replay_buffer.tree.capacity - 1
    replay_buffer.update_priorities(td_error)


N = 1000000
N_TRAIN_CRITIC = 10
N_TRAIN_TRUE_CRITIC = 10000
SAMPLES = 3
def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    agent._replay_buffer.load_data('./per_exp/eval_grads/buffer_data/')
    accum = Accumulator()

    buff = agent._replay_buffer
    buff.size = N 
    buff.ptr = N
    buff.max_size = N
    buff.tree.n_entries = N
    untrained = copy.deepcopy(agent)
    agent._policy.actor.load_weights('./per_exp/eval_grads/model/converged_actor')

    trained_actor = agent._policy.actor
    print('Successfully loaded')
    print('Compute true gradient of true critic')
    # True gradient of true critic 
    for i in range(SAMPLES):
        true_critic_sample = copy.deepcopy(untrained)
        train_the_critic(true_critic_sample, buff, N_TRAIN_TRUE_CRITIC)
        true_critic_sample = true_critic_sample._policy.critic
        state, *_ = buff.get_buffer()
        with tf.GradientTape() as tape:
            action = trained_actor(state)
            q_value, _  = true_critic_sample(tf.concat([state, action], axis=1))
            actor_loss = -tf.reduce_mean(q_value)
        gradients_true = tape.gradient(actor_loss, trained_actor.trainable_variables)
        gradients_true  = [tf.reshape(x, [-1]) for x in gradients_true]
        accum.accumulate(gradients_true)
    gradients_true = accum.get_grad() 

    # TODO AVERAGE OVER BATCHES
    batch_range = np.array([1, 64, 128, 256, 1000])#, np.arange(1000, 6000, 1000)], axis=0)
    simil_list = []
    for batch_size in batch_range:
        print(f'Batch {batch_size}')
        accum.reset()
        sims_collect = 0
        for i in range(SAMPLES):
            approx_critic = copy.deepcopy(untrained)
            train_the_critic(approx_critic, buff, N_TRAIN_CRITIC)
            print('Update Buffer')
            update_buffer(buff, approx_critic)
            approx_critic = approx_critic._policy.critic
            state, *_ = buff.sample(batch_size)
            with tf.GradientTape() as tape:
                action = trained_actor(state)
                q_value, _  = approx_critic(tf.concat([state, action], axis=1))
                actor_loss = -tf.reduce_mean(q_value)
            gradients_sample = tape.gradient(actor_loss, trained_actor.trainable_variables)
            gradients_sample = [tf.reshape(x, [-1]) for x in gradients_sample]
            sims = [-simil_metric(x, y) for x, y in zip(gradients_true, gradients_sample)]
            sims = tf.reduce_mean(sims)
            set_trace()
            sims_collect += sims.numpy()
        sims_collect /= SAMPLES
        simil_list.append(sims_collect)
        print(simil_list)
    print(simil_list)
    np.save('simil_list.npy', simil_list)





    #np.save('m1.npy', m1)    
    #np.save('m2.npy', m2)    
    #np.save('errors.npy', errors)
    env.close()

