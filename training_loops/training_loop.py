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

class Reset_Reversal:
    def __init__(self, agent, N, active=True):
        self.tmp = False
        self.N = N
        self.agent = agent
        self.active = active
        self.its = 0

    def maybe_reset_things_for_reversal(self, t):
        if t == self.N and self.active:
            #self.agent.meta_replay_buffer.reset()
            #self.agent._meta_agent.beta_1.assign(0)
            #self.agent._meta_agent.beta_2.assign(0)
            #self.agent._meta_agent.critic_optimizer.iterations.assign(0)
            #self.agent._meta_agent.actor_optimizer.iterations.assign(0)
            self.old = self.agent._meta_agent.actor_optimizer.learning_rate.numpy()
            self.agent._meta_agent.actor_optimizer.learning_rate.assign(0.0008)
            #self.agent._meta_agent.full_reset()
            #self.agent._meta_noise = 3.5
            #self.tmp = True
            self.its += 1
        if t > self.N and self.tmp == True:
            if self.its >= 10000:
                self.agent._meta_agent.actor_optimizer.learning_rate = self.old
            #self.agent._meta_agent.beta_1.assign(0.9)
            #self.agent._meta_agent.beta_2.assign(0.999)

            #self.tmp = False

def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    # create objects 
    logger = Logger(cnf.log, cnf.minilog, env.time_limit)
    # Load previously trained model.
    if cnf.load_model: agent.load_model(f'./experiments/models/{agent._load_string}')
    # Training loop
    state, done = env.reset(), False
    switch = 0
    reward_fn = tf.Variable(0)
    for t in range(int(cnf.max_timesteps)):
        if not t % cnf.switch_time:
            switch = (switch + 1) % 2
        action = agent.select_action(state, noise_bool=True, reward_fn=reward_fn)
        next_state, reward, done, _ = env.step(action, reward_fn)
        # future value fct only zero if terminal because of success, not time
        success_cd = [done if env.success else 0][0]
        # get intrinsic reward from agent.transitbuffer computation
        intr_rew = agent.replay_add(state, action, reward, next_state, done, success_cd, FM=None)
        maybe_verbose_output(t, agent, env, action, cnf, state, intr_rew)
        logger.inc(t, reward)
        if not cnf.flat_agent and not cnf.minilog:
            logger.most_important_plot(agent, state, action, reward, next_state, success_cd)
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

