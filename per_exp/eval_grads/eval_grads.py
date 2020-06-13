import numpy as np
import tensorflow as tf
from rl_algos.TD3_tf import Actor
import wandb
import copy
from utils.utils import create_world, create_agent
from matplotlib import pyplot as plt
simil_metric = tf.keras.losses.CosineSimilarity()

N = 1000000
N_TRAIN_CRITIC = 10
N_TRAIN_TRUE_CRITIC = 100000
SAMPLES = 50
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

def train_the_critic(untrained_agent, replay_buffer, iterations, batchsize):
    for it in range(iterations):
        print(f'{it} of {iterations} training')
        state, action, reward, next_state, done, *_ = replay_buffer.sample_uniformly(batchsize)
        td_error = untrained_agent._policy._train_critic(state, action, reward, next_state, done, False, None)
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

    accum = Accumulator()
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
    for i in range(SAMPLES):
        set_seeds(i)
        print(f'sample {i} of {SAMPLES} Highqualitycritic')
        true_critic_sample = create_agent(cnf, env)
        train_the_critic(true_critic_sample, buff, N_TRAIN_TRUE_CRITIC, 128)
        true_critic_sample = true_critic_sample._policy.critic
        state, *_ = buff.get_buffer()
        with tf.GradientTape() as tape:
            action = trained_actor(state)
            q_value, _  = true_critic_sample(tf.concat([state, action], axis=1))
            actor_loss = -tf.reduce_mean(q_value)
        gradients_true = tape.gradient(actor_loss, trained_actor.trainable_variables)
        gradients_true  = [tf.reshape(x, [-1]) for x in gradients_true]
        gradients_true = tf.concat(gradients_true, axis=0)
        accum.accumulate(gradients_true)
    grad_mean, grad_std = accum.get_grad() 
    batch_range = np.array([1, 32, 64, 128, 256, 512, 1024, 2048])
    simil_values= []

    accum2 = Accumulator()
    for batch_size in batch_range:
        print(f'Batch {batch_size}')
        similarity_samples = []
        for i in range(SAMPLES):
            accum.reset()
            set_seeds(i+SAMPLES)
            print(f'sample {i} of {SAMPLES} lowqualitycritic')
            approx_critic = create_agent(cnf, env)
            train_the_critic(approx_critic, buff, N_TRAIN_CRITIC, 128)
            print('Update Buffer')
            #update_buffer(buff, approx_critic)
            approx_critic = approx_critic._policy.critic

            for i in range(BATCHES):
                accum2.reset()
                state, *_ = buff.sample_uniformly(batch_size)
                with tf.GradientTape() as tape:
                    action = trained_actor(state)
                    q_value, _  = approx_critic(tf.concat([state, action], axis=1))
                    actor_loss = -tf.reduce_mean(q_value)
                gradients_sample = tape.gradient(actor_loss, trained_actor.trainable_variables)
                gradients_sample = [tf.reshape(x, [-1]) for x in gradients_sample]
                gradients_sample = tf.concat(gradients_sample, axis=0)
                accum2.accumulate(gradients_sample)
            # averaged over batches
            avg_grad, _ = accum2.get_grad()
            sample_similarity = -simil_metric(avg_grad, grad_mean)
            similarity_samples.append(sample_similarity)
        simil_values.append(similarity_samples)
        print(np.mean(simil_values[-1]))
    print(np.mean(simil_values[-1]))
    np.save('simils_unif.npy', simil_values)





    #np.save('m1.npy', m1)    
    #np.save('m2.npy', m2)    
    #np.save('errors.npy', errors)
    env.close()

