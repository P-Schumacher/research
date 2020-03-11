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

def maybe_verbose_output(t, agent, env, action, cnf, state):
    if cnf.render:
        print(f"action: {action}")
        print(f"time is: {t}")
        print(f"state is {state}")
        if not cnf.flat_agent:
            print(f"goal: {agent.goal}")
            if agent.meta_time and cnf.render:
                print(f"GOAL POSITION: {agent.goal}")
                env.set_goal(agent.goal[:3])

def decay_step(decay, stepper, agent):
    c_step = 1
    if decay:
        c_step = int(next(stepper))
        agent.goal_every_n = c_step
        agent.c_step = c_step
        agent.meta_agent.c_step = c_step
    return c_step


def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    # create objects 
    logger = Logger(cnf.log, cnf.time_limit)
    stepper = exponential_decay(**cnf.step_decayer)
    # Load previously trained model.
    if cnf.load_model: agent.load_model(f'./models/{agent._file_name}')
    # Training loop
    state, done = env.reset(), False
    for t in range(int(cnf.max_timesteps)):
        c_step = decay_step(cnf.decay, stepper, agent)
        state[:3] = 0
        state[10:12] = 0
        action = agent.select_noisy_action(state)
        maybe_verbose_output(t, agent, env, action, cnf, state)
        next_state, reward, done, _ = env.step(action)
        intr_rew = agent.replay_add(state, action, reward, next_state, done)
        if t > cnf.start_timesteps and not t % cnf.train_every:
            agent.train(t)
        state = next_state
        logger.inc(t, reward)

        if done:
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1}+ Episode T: {logger.episode_timesteps} Reward: {logger.episode_reward}")
            # Reset environment
            agent.reset()
            hard_reset = logger.log(t, intr_rew, c_step)
            logger.reset()
            if hard_reset:
                # Need to periodically restart physics engine because Darmstadt gripper model is unstable.
                # Hard reset takes current arm position as initial position. This is why we first do a normal reset.
                env.reset()
            state, done = env.reset(hard_reset=hard_reset), False
        # Evaluate episode
        if (t + 1) % cnf.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate = agent.evaluation(env)
            state, done = env.reset(), False
            agent.reset()
            logger.reset(post_eval=True)
            logger.log_eval(t, avg_ep_rew, avg_intr_rew, success_rate)
            if cnf.save_model: agent.save_model(f'./models/action_regul_{agent._file_name}')

