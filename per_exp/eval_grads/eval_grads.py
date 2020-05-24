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

def create_random_weight_list():
    weights = []
    for i in range(6):
        weights.append(tf.random.uniform([20, 300], -2, 2))
    return weights

def train_the_critic(untrained_agent, replay_buffer, iterations):
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00075)
    for it in range(iterations):
        state, action, reward, next_state, done, *_ = replay_buffer.sample_uniformly(128)
        untrained_agent._policy._train_step(state, action, reward, next_state, done, False, None)
        

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
    untrained = copy.deepcopy(agent)
    agent._policy.actor.load_weights('./per_exp/eval_grads/model/converged_actor')
    agent._policy.critic.load_weights('./per_exp/eval_grads/model/converged_critic')

    trained_actor = agent._policy.actor
    true_critic = agent._policy.critic


    print('Successfully loaded')
    train_the_critic(untrained, buff, 2)
    approx_critic = untrained._policy.critic
    set_trace()
    # True gradient of true critic 
    state, *_ = buff.sample_uniformly(N_true)
    with tf.GradientTape() as tape:
        action = trained_actor(state)
        q_value, _  = true_critic(tf.concat([state, action], axis=1))
        actor_loss = -tf.reduce_mean(q_value)
    gradients_true = tape.gradient(actor_loss, trained_actor.trainable_variables)
    gradients_true  = [tf.reshape(x, [-1]) for x in gradients_true]
    
    simil_list = []
    # gradient of approximated critic
    for batch_size in np.arange(2, 10000, 200):
        print(f'batch size {batch_size} of {N_true}')
        state, *_ = buff.sample_uniformly(batch_size)
        with tf.GradientTape() as tape:
            action = trained_actor(state)
            q_value, _  = approx_critic(tf.concat([state, action], axis=1))
            actor_loss = -tf.reduce_mean(q_value)
        gradients_sample = tape.gradient(actor_loss, trained_actor.trainable_variables)
        gradients_sample  = [tf.reshape(x, [-1]) for x in gradients_sample]

        sims = [-simil_metric(x, y) for x, y in zip(gradients_true, gradients_sample)]
        sims = tf.reduce_mean(sims)
        simil_list.append(sims.numpy())
    np.save('simil_list.npy', np.asarray(simil_list))





    #np.save('m1.npy', m1)    
    #np.save('m2.npy', m2)    
    #np.save('errors.npy', errors)
    env.close()

