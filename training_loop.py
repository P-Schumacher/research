from matplotlib import pyplot as plt
import numpy as np
import gym
import argparse
import os
import sys
from agent_files.HIRO import HierarchicalAgent
from utils.utils import create_world, setup,  exponential_decay
import time
import math
import tensorflow as tf
from pudb import set_trace
import datetime
import wandb

class Logger:
    def __init__(self):
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0

    def inc(self, reward):
        self.episode_timesteps += 1
        self.episode_reward += reward

    def reset(self, post_eval=False):
        self.episode_timesteps = 0
        self.episode_reward = 0
        if not post_eval:
            self.episode_num += 1
    
    def log(self, logging, intr_rew):
        if logging:
            wandb.log({'ep_rew': self.episode_reward, 'intr_reward': intr_rew})


def maybe_verbose(agent, action, args):
    if args.render:
        print("action: " + str(action))
        print("time is: " + str(t))
        if not args.flat_agent:
            print("goal: " + str(agent.goal))
            if agent.meta_time and args.render:
                print("GOAL POSITION: " + str(agent.goal))
                env.set_goal(agent.goal[:3])

if __name__ == "__main__":
    # Parse Arguments and create directories
    args = setup(sys.argv[1:])
    if args.wandb:
        wandb.init(project='research', entity='rlpractitioner', config=args)
    # create environment and agent
    env, agent = create_world(args)
    # Load previously trained model.
    if args.load_model: agent.load_model("./models/" + str(agent.file_name))
    # Create logger
    logger = Logger()
    # Start env
    state, done = env.reset(), False
    # Use tf Variable here because we want to use it in static graph later
    time_step = tf.Variable(0, dtype=tf.int64)
    # Training loop
    for t in range(int(args.max_timesteps)):
        if t < args.start_timesteps:
            action = agent.random_action(state) 
        else:
            action = agent.select_noisy_action(state)
        maybe_verbose(agent, action, args)
        next_state, reward, done, _ = env.step(action)
        # Store data in replay buffer
        intr_rew = agent.replay_add(state, action, reward, next_state, done)
        if t > args.start_timesteps:
            agent.train(time_step)
       
        state = next_state
        logger.inc(reward)
        time_step.assign_add(1)
        
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1}+ Episode T: {logger.episode_timesteps} Reward: \
                  {logger.episode_reward}")
            # Reset environment
            state, done = env.reset(), False
            agent.reset()
            logger.log(args.wandb, intr_rew)
            logger.reset()
        
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate = agent.evaluation(env)
            state, done = env.reset(), False
            agent.reset()
            logger.reset(post_eval=True)
            if args.wandb:
                wandb.log({'eval/eval_ep_rew': avg_ep_rew, 'eval/eval_intr_rew': avg_intr_rew,
                      'eval/success_rate': success_rate})
            if args.save_model: agent.save_model("./models/"+str(agent.file_name))
