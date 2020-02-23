from matplotlib import pyplot as plt
import numpy as np
import gym
import argparse
import os
import sys
from utils.utils import create_world, setup
from agent_files.HIRO import HierarchicalAgent
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820 as Kuka
import tensorflow as tf
from pudb import set_trace

# python my_main.py --env AntMaze --policy TD3 --start_timesteps 10000 --load_model default --save_model --render 
META_RANGES = np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3], dtype=np.float32)
mock_goal = np.copy(META_RANGES)
mock_goal[0] = 10
mock_goal[1] = 1

def init_kuka():
    robot = Kuka(count=1)
    robot.set_control_loop_enabled(True)
    return robot

def set_kuka(robot, pos):
    robot.set_joint_target_positions(pos)
    return 

def random_goal_action(agent, state):
	agent.select_action(state)
	if agent.transitbuffer.meta_time:
		agent.transitbuffer.goal = np.random.uniform(-2, 1, META_RANGES.shape[0])
		agent.transitbuffer.goal *=  META_RANGES
	action = agent.sub_agent.select_action(np.concatenate([state, agent.transitbuffer.goal]))
	return action

def mock_goal_action(agent, state):
	agent.select_action(state)
	if agent.transitbuffer.meta_time:
		agent.transitbuffer.goal = mock_goal
	action = agent.sub_agent.select_action(np.concatenate([state, agent.transitbuffer.goal]))
	return action

if __name__ == "__main__":
    # Parse Arguments and create directories
    args = setup(sys.argv[1:])
    env, agent = create_world(args)
    #agent.load_model(f"./EXPERIMENTS/directional_10_0_1M/model/{agent.file_name}")
    #agent.load_model(f"./EXPERIMENTS/absolute_10_0/model/{agent.file_name}")
    #agent.load_model(f"./EXPERIMENTS/absolute_left_right_10/model/{agent.file_name}")
    #agent.load_model(f"./EXPERIMENTS/6m_absolute_left_right_up_down/{agent.file_name}")
    #agent.load_model(f"./EXPERIMENTS/2m_full_hiro_agent_15_5/model/{agent.file_name}")
    agent.load_model(f"./models/{agent.file_name}")
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    second_bot = init_kuka()
    init_state = state
    while True:
        second_bot.set_joint_positions(init_state[:7])
        state = env.reset()
        print('Initial position: ') 
        print(env._init_pos)
        for t in range(int(100000)):
            episode_timesteps += 1
            action = agent.select_action(state)  # Full HIRO
            print(action)
            #action = random_goal_action(agent, state)  # Random Goal
            #action = mock_goal_action(agent, state)  # Constant Goal defined in beginning
            if agent.meta_time:
                if args.ee_pos:
                    env.set_goal(agent.goal[:3])
                else:
                    set_kuka(second_bot, agent.goal)
            state, reward, done, _ = env.step(action)
            print(f"REWARD:  {reward}")
            print(f"DONE: {done}")
            print(f" t = {t}")
            print(f"distance from goal {np.linalg.norm(state[:agent.goal.shape[0]] - agent.goal)}")
            episode_reward += reward
            if done: 
                print(f'episode reward is: {episode_reward}')
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                
