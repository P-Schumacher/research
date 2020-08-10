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
from utils.utils import create_world, exponential_decay
from rl_algos.FM import ForwardModel


def maybe_verbose_output(t, agent, env, action, cnf, state, reward):
    if cnf.render:
        if not cnf.flat_agent:
            if cnf.render:
                if agent._smooth_goal:
                    goal = agent._prev_goal
                else:
                    goal = agent.goal
                if agent.goal_type == 'Direction' or agent.goal_type == 'Sparse':
                    env.set_goal(state[:3] + goal[:3])
                else:
                    env.set_goal(goal[:3])

def decay_step(decay, stepper, agent, flat_agent, init_c):
    c_step = [1 if flat_agent else init_c][0]
    if decay:
        c_step = int(next(stepper))
        agent._c_step = c_step
        agent._meta_agent.c_step = c_step
    return c_step

class Reset_Reversal:
    def __init__(self, agent, N, active=False):
        self.tmp = False
        self.N = N
        self.agent = agent
        self.active = active

    def maybe_reset_things_for_reversal(self, t):
        if t == self.N and self.active:
            self.agent.meta_replay_buffer.reset()
            self.agent._meta_agent.beta_1.assign(0)
            self.agent._meta_agent.beta_2.assign(0)
            self.agent._meta_agent.critic_optimizer.iterations.assign(0)
            self.agent._meta_agent.actor_optimizer.iterations.assign(0)
            self.agent._meta_agent.full_reset()
            self.tmp = True
        if t > self.N and self.tmp == True:
            self.agent._meta_agent.beta_1.assign(0.9)
            self.agent._meta_agent.beta_2.assign(0.999)
            self.tmp = False

def main(cnf):
    reversal = False
    env, agent = create_world(cnf)
    #reverser = Reset_Reversal(agent, cnf.coppeliagym.params.reversal_time)
    FM = ForwardModel(26, logging=cnf.main.log, oracle=False)
    cnf = cnf.main
    # create objects 
    logger = Logger(cnf.log, cnf.minilog, cnf.time_limit)
    stepper = exponential_decay(**cnf.step_decayer)
    # Load previously trained model.
    if cnf.load_model: agent.load_model(f'./experiments/models/{agent._file_name}')
    # Training loop
    state, done = env.reset(), False
    run_online = 0.
    for t in range(int(cnf.max_timesteps)):
        #reverser.maybe_reset_things_for_reversal(t)
        c_step = decay_step(cnf.decay, stepper, agent, cnf.flat_agent, cnf.c_step)
        action = agent.select_action(state, noise_bool=True)
        next_state, reward, done, _ = env.step(action)
        # future value fct only zero if terminal because of success, not time
        success_cd = [done if env.success else 0][0]
        intr_rew = agent.replay_add(state, action, reward, next_state, done, success_cd, FM)
        FM.train(state, next_state, reward, success_cd, done)
        maybe_verbose_output(t, agent, env, action, cnf, state, intr_rew)
        logger.inc(t, reward)
        onlineprederr = tf.abs(reward - FM.forward_pass(tf.reshape(state, [1,26]), tf.reshape(next_state, [1,26]))[0])
        run_online = 0.2 * run_online + 0.8 * onlineprederr
        if cnf.log:
            wandb.log({f'FM/avgonlineprederr':run_online}, commit=False)
        #logger.most_important_plot(agent, state, action, reward, next_state, success_cd)
        state = next_state
        if t == 200000:
            FM = ForwardModel(26, logging=cnf.log, oracle=False)
        if done:
            # Train at the end of the episode for the appropriate times. makes collecting
            # norms stds and losses easier
            if t > cnf.start_timesteps:
                agent.train(t, logger.episode_timesteps, FM)
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1} Episode T: {logger.episode_timesteps} Reward: {logger.episode_reward}")
            logger.log(t, intr_rew, c_step)
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
            if cnf.save_model: agent.save_model(f'./experiments/models/{agent._file_name}')

