import numpy as np
import tensorflow as tf
from rl_algos.TD3_tf import Actor
import wandb
import copy
from utils.utils import create_world, create_agent
from matplotlib import pyplot as plt
simil_metric = tf.keras.losses.CosineSimilarity()
from pudb import set_trace

N = 1000
N_TRAIN_CRITIC = 100
N_TRAIN_TRUE_CRITIC = 1000
SAMPLES = 50
SAMPLES_TRUE_CRITIC = 50
BATCHES = 10

class Accumulator:
    def __init__(self):
        self.grad = None
        self.init = False

    def accumulate(self, grad):
        if not self.init:
            self.grad = tf.reshape(grad, [1, grad.shape[0]])
            self.init = True
        else:
            self.grad = tf.concat([self.grad, tf.reshape(grad, [1, grad.shape[0]])], axis=0)

    def get_grad(self):
        return np.mean(self.grad.numpy(), axis=0), np.std(self.grad.numpy(), axis=0)

    def reset(self):
        self.grad = None
        self.init = False

class Accumulator2:
    def __init__(self):
        self.grad = None
        self.init = False
        self.counter = 0
    
    def accumulate(self, grad):
        self.counter += 1
        if not self.init:
            self.grad = grad
            self.init = True
        else:
            self.grad = [x+y for x,y in zip(self.grad, grad)]

    def get_grad(self):
        return [x / self.counter for x in self.grad]

    def reset(self):
        self.grad = None
        self.counter = 0
        self.init = False
        

def train_the_critic(untrained_agent, replay_buffer, iterations, batchsize):
    for it in range(iterations):
        print(f'{it} of {iterations} training')
        state, action, reward, next_state, done, *_ = replay_buffer.sample(batchsize)
        td_error = untrained_agent._policy._train_critic(state, action, reward, next_state, done, False, None)
        replay_buffer.update_priorities(tf.abs(td_error))
        #wandb.log({'tderror': tf.reduce_mean(tf.abs(td_error)).numpy()})

def update_buffer(replay_buffer, agent):
    state, action, reward, next_state, done = replay_buffer.get_buffer()
    td_error = agent._policy._compute_td_error(state, action, reward, next_state, done)
    replay_buffer.ptr = 0
    replay_buffer.tree.write = 0
    for x in td_error:
        replay_buffer.add_just_priority(x)
        replay_buffer.ptr += 1

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def main(cnf):
    env, agent = create_world(cnf)
    agent._replay_buffer.load_data('./per_exp/eval_grads/buffer_data/', N)

    accum = Accumulator2()
    buff = agent._replay_buffer
    buff.size = N 
    buff.ptr = 0
    buff.tree.write = 0
    buff.max_size = N
    buff.tree.n_entries = N
    untrained = copy.deepcopy(agent)
    agent._policy.actor.load_weights('./per_exp/eval_grads/model/converged_actor')

    trained_actor = agent._policy.actor
    print('Successfully loaded')
    print('Compute true gradient of true critic')
    # True gradient of true critic 
    for i in range(SAMPLES_TRUE_CRITIC):
        set_seeds(i)
        print(f'sample {i} of {SAMPLES_TRUE_CRITIC} Highqualitycritic')
        true_critic_sample = create_agent(cnf, env)
        train_the_critic(true_critic_sample, buff, N_TRAIN_TRUE_CRITIC, 128)
        true_critic_sample = true_critic_sample._policy.critic
        gradient_state_list = [] 
        for idx, state in enumerate(buff.state):
            print(f' {idx} of {N}')
            state = tf.reshape(state, [1, state.shape[0]])
            with tf.GradientTape() as tape:
                action = trained_actor(state)
                q_value, _  = true_critic_sample(tf.concat([state, action], axis=1))
                actor_loss = -tf.reduce_mean(q_value)
            gradients_true = tape.gradient(actor_loss, trained_actor.trainable_variables)
            gradients_true = [tf.reshape(x, [-1]) for x in gradients_true]
            gradients_true = tf.concat(gradients_true, axis=0)
            gradient_state_list.append(gradients_true)
        accum.accumulate(gradient_state_list)
    grad_true_per_element = accum.get_grad() 
    
    scatter_metric = []
    scatter_td_error = []
    accum.reset()
    accu_td = Accumulator()
    for i in range(SAMPLES):
        set_seeds(i+SAMPLES)
        print(f'sample {i} of {SAMPLES} lowqualitycritic')
        approx_critic = create_agent(cnf, env)
        train_the_critic(approx_critic, buff, N_TRAIN_CRITIC, 128)
        print('Update Buffer')
        update_buffer(buff, approx_critic)
        approx_critic = approx_critic._policy.critic
        gradient_state_list_lq = []
        for idx, state in enumerate(buff.state):
            print(f' {idx} of {N}')
            state = tf.reshape(state, [1, state.shape[0]])
            with tf.GradientTape() as tape:
                action = trained_actor(state)
                q_value, _  = approx_critic(tf.concat([state, action], axis=1))
                actor_loss = -tf.reduce_mean(q_value)
            gradients_sample = tape.gradient(actor_loss, trained_actor.trainable_variables)
            gradients_sample = [tf.reshape(x, [-1]) for x in gradients_sample]
            gradients_sample = tf.concat(gradients_sample, axis=0)
            gradient_state_list_lq.append(gradients_sample)
        accum.accumulate(gradient_state_list_lq)
        accu_td.accumulate(buff.priorities.copy())
        
    grad_sample = accum.get_grad()
    priorities, _ = accu_td.get_grad()
    metric = [-simil_metric(x, y) for x,y in zip(grad_true_per_element, grad_sample)]

    np.save('scatter_metric.npy', metric)
    np.save('scatter_td_error.npy', priorities)





    #np.save('m1.npy', m1)    
    #np.save('m2.npy', m2)    
    #np.save('errors.npy', errors)
    env.close()

