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

def maybe_verbose_output(t, agent, env, action, cnf, state, reward):
    if cnf.render:
        if not cnf.flat_agent:
            if agent.meta_time and cnf.render:
                print(f't: {t}')
                #print(f"GOAL POSITION: {agent.goal}")
                if agent.goal_type == 'Direction':
                    env.set_goal(state[:3] + agent.goal[:3])
                else:
                    env.set_goal(agent.goal[:3])

def decay_step(decay, stepper, agent, flat_agent):
    c_step = [1 if flat_agent else 10][0]
    if decay:
        c_step = int(next(stepper))
        agent._c_step = c_step
        agent._meta_agent.c_step = c_step
    return c_step


def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    # create objects 
    logger = Logger(cnf.log, cnf.time_limit)
    stepper = exponential_decay(**cnf.step_decayer)
    # Load previously trained model.
    if cnf.load_model: agent.load_model(f'./experiments/models/{agent._file_name}')
    # Training loop
    state, done = env.reset(), False
    for t in range(int(cnf.max_timesteps)):
        c_step = decay_step(cnf.decay, stepper, agent, cnf.flat_agent)
        action = agent.select_noisy_action(state)
        next_state, reward, done, _ = env.step(action)
        intr_rew = agent.replay_add(state, action, reward, next_state, done)
        maybe_verbose_output(t, agent, env, action, cnf, state, intr_rew)
        state = next_state
        logger.inc(t, reward)

        if done:
            if t > cnf.start_timesteps:
                agent.train(t, logger.episode_timesteps)
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1}+ Episode T: {logger.episode_timesteps} Reward: {logger.episode_reward}")
            logger.log(t, intr_rew, c_step)
            # Reset environment
            agent.reset()
            logger.reset()
            state, done = env.reset(), False
        # Evaluate episode
        if (t + 1) % cnf.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate = agent.evaluation(env)
            state, done = env.reset(), False
            agent.reset()
            logger.reset(post_eval=True)
            logger.log_eval(t, avg_ep_rew, avg_intr_rew, success_rate)
            if cnf.save_model: agent.save_model(f'./experiments/models/{agent._file_name}')

