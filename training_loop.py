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

timer = datetime.datetime.now()
writer = tf.summary.create_file_writer("./runs/main" + str(timer.minute))

def maybe_render(agent, action, args):
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
    # create environment and agent
    env, agent = create_world(args)
    # Load previously trained model.
    if args.load_model: agent.load_model("./models/" + str(agent.file_name))
    # Start env
    state, done = env.reset(), False
    # Reset counters
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    # Use tf Variable here because we want to use it in static graph later
    time_step = tf.Variable(0, dtype=tf.int64)
    start = time.time()
    # Training loop
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if t < args.start_timesteps:
            action = agent.random_action(state) 
        else:
            action = agent.select_noisy_action(state)
        maybe_render(agent, action, args)
        next_state, reward, done, _ = env.step(action)
        # Store data in replay buffer
        agent.replay_add(state, action, reward, next_state, done)
        if t > args.start_timesteps:
            agent.train(time_step)
       
        state = next_state
        episode_reward += reward
        time_step.assign_add(1)
        
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print("Total T: "+str(t+1)+"Episode Num: "+str(episode_num+1) +"Episode T:"
                  +str(episode_timesteps)+"Reward: " +str(episode_reward))
            # Reset environment
            state, done = env.reset(), False
            agent.reset()
            with writer.as_default():
                tf.summary.scalar("data/ep_reward", episode_reward, t)
            
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
        
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate = agent.evaluation(env)
            #np.save(f"./results/{agent.file_name}", avg_ep_reward)
            with writer.as_default():
                tf.summary.scalar("eval/eval_ep_rew", avg_ep_rew, t)
                tf.summary.scalar("eval/eval_intr_rew", avg_intr_rew, t)
                tf.summary.scalar("eval/success_rate", success_rate, t)
            if args.save_model: agent.save_model("./models/"+str(agent.file_name))
    print(time.time() - start)
