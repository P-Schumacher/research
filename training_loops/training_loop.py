import numpy as np
import gym
import sys
import time
import math
import tensorflow as tf
import wandb
from pudb import set_trace
from agent_files.HIRO import HierarchicalAgent
from utils.logger import Logger
from utils.utils import create_world, exponential_decay, decay_step, maybe_verbose_output
from rl_algos.FM import ForwardModel

def concat(a,b):
    assert a.shape[0] == 1 and b.shape[0] == 1
    c = np.zeros([a.shape[0], a.shape[-1]+b.shape[-1]], dtype=np.float32)
    c[:, :a.shape[-1]] = a
    c[:, :b.shape[-1]] = b
    return c 

def attention_grad_meta(state, agent, t, flat):
    c_step = [1 if flat else 10][0]
    state = tf.reshape(state, shape=[1, state.shape[0]])
    if not flat:
        goal = tf.reshape(agent.goal, shape=[1, agent.goal.shape[0]])
        grads = agent._meta_agent._get_attention_gradients_cr(state, goal)
        sub_state = concat(state[:, :-6], goal)
        action = agent._sub_agent.actor(sub_state)
        grads_sub = agent._sub_agent._get_attention_gradients_cr(sub_state, action)
        grads_sub = grads_sub[:grads.shape[0]]
    else:
        action = agent._policy.actor(state)
        grads = agent._policy._get_attention_gradients_cr(state, action)
    np.save(f'./grad_attention/c{c_step}/grad_critic_meta_{t}.npy',np.array(grads))
    state_padded = np.zeros_like(grads)
    state_padded[:state.shape[1]] = state
    np.save(f'./grad_attention/c5/grad_critic_meta_{t}.npy',state_padded)
    if not flat:
        np.save(f'./grad_attention/c33/grad_critic_meta_{t}.npy', grads_sub)



def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    FM = ForwardModel(26, cnf.log, oracle=False)
    # create objects 
    logger = Logger(cnf.log, cnf.minilog, env.time_limit)
    # Load previously trained model.
    if cnf.load_model: agent.load_model(f'./experiments/models/{agent._load_string}')
    # Training loop
    state, done = env.reset(), False
    switch = 0
    reward_fn = tf.Variable(0)
    for t in range(int(cnf.max_timesteps)):
        state_old = state
        if not t % cnf.switch_time:
            switch = (switch + 1) % 2
        action = agent.select_action(state, noise_bool=True, reward_fn=reward_fn)
        next_state, reward, done, _ = env.step(action, reward_fn)
        success_cd = [done if env.success else 0][0]
        # get intrinsic reward from agent.transitbuffer computation
        intr_rew = agent.replay_add(state, action, reward, next_state, done, success_cd)
        maybe_verbose_output(t, agent, env, action, cnf, state, intr_rew)
        logger.inc(t, reward)
        if not cnf.flat_agent and not cnf.minilog:
            logger.most_important_plot(agent, state, action, reward, next_state, success_cd)
        if cnf.save_attention:
            attention_grad_meta(state, agent, t, flat=cnf.flat_agent)
        FM.train(state, next_state, reward, success_cd, done)
        state = next_state
        if done:
            reward_fn.assign([0 if switch else 1][0])
            # Train at the end of the episode for the appropriate times. makes collecting
            # norms stds and losses easier
            if t > cnf.start_timesteps:
                agent.train(t, logger.episode_timesteps)
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1} Episode T: {logger.episode_timesteps} Reward: {logger.episode_reward}")
            logger.log(t, intr_rew)
            agent.reset()
            logger.reset()
            state, done = env.reset(), False
        # Evaluate episode
        if (t + 1) % cnf.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate, rate_correct_solves, untouchable_steps = agent.evaluation(env)
            state, done = env.reset(), False
            agent.reset()
            logger.reset(post_eval=True)
            logger.log_eval(t, avg_ep_rew, avg_intr_rew, success_rate, rate_correct_solves, untouchable_steps)
            if cnf.save_model: agent.save_model(f'./experiments/models/{agent._load_string}')

