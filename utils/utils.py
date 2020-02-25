import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from copy import deepcopy
from pudb import set_trace
from collections import deque
from agent_files import TD3_tf
from environments.my_env import EnvWithGoal
from agent_files.Agent import Agent
from agent_files.HIRO import HierarchicalAgent

def get_policy_args(args, env_specs, subgoal_ranges, target_dim):
    '''Creates arguments for the policy class from the global arguments and the environment.'''
    meta_args =  {
    "state_dim": env_specs[0],
    "action_dim": env_specs[1],
    "max_action": env_specs[2],
    "discount": args.discount,
    "tau": args.tau,
    "policy_noise": args.policy_noise,
    "noise_clip": args.noise_clip,
    "policy_freq": args.policy_freq,
    "subgoal_ranges": subgoal_ranges,
    "c_step": args.c_step,
    "target_dim": target_dim,
    "actr_lr": args.meta_actr_lr,
    "ctr_lr": args.meta_ctr_lr,
    "offpolicy": args.offpolicy,
    "name": "meta"
    }
    sub_args = deepcopy(meta_args)
    sub_args["actr_lr"] = args.sub_actr_lr
    sub_args["ctr_lr"] = args.sub_ctr_lr
    sub_args["offpolicy"] = False
    sub_args["name"] = "sub"
    return meta_args, sub_args

def get_model_class(model_name):
	if model_name == "TD3":
		return TD3_tf.TD3

def create_env(args):
    '''Creates an environment from either OpenAi Gym or the GoogleBrain Mujoco AntMaze
    gym environment.
    It wraps it such that it tries to reach a global target position,
    which is appended to the obs. It takes obs[:2] as x,y coordinates.
    '''
    if args.vrep:
        # Load cool robotics env
        from environments.coppeliagym import CoppeliaEnv 
        print("Force mode is "+str(args.force))
        print("ee pos is: "+str(args.ee_pos))
        env = CoppeliaEnv(args)
        print("Target is: "+str(env._ep_target_pos))
        if args.render:
            env.render()
        return env
    else:
        # *show* necessary because we need to load a different xml file with spheres
        from environments.create_maze_env import create_maze_env
        env = create_maze_env(args.env, show=args.render)
        return EnvWithGoal(env, args.env_name, args.time_limit, render=args.render, evalmode=False)

def get_env_specs(env):
        ''' Get necessary dimensions from environment to instantiate model.'''
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        if env.action_space.high.shape[0] > 1:
            max_action = env.action_space.high
        else: 
            max_action = float(env.action_space.high[0])
        time_limit = env._max_episode_steps
        target_dim = env.target_dim
        subgoal_ranges = env.subgoal_ranges
        subgoal_dim = env.subgoal_dim
        target_dim = env.target_dim
        print("Env Specs: ")
        print(f"state_dim: {state_dim}")
        print(f"action_dim: {action_dim}")
        print(f"max_action: {max_action}")
        print(f"time_limit: {time_limit}")
        print(f"subgoal_dim: {subgoal_dim}")
        print(f"target_dim: {target_dim}")
        return (state_dim, action_dim, max_action, time_limit), subgoal_dim, subgoal_ranges, target_dim

def create_directories(args):
    '''Create directories to save weights and results.'''
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

def set_seeds(env, seed):
    '''Set seeds to get different random numbers for every experiment. Seeds have to be set by CMD Prompt.'''
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def create_world(args):
    create_directories(args)
    env = create_env(args)
    model = get_model_class(args.policy)
    env_specs, subgoal_dim, subgoal_ranges, target_dim = get_env_specs(env)
    meta_args, sub_args = get_policy_args(args, env_specs, subgoal_ranges, target_dim)
    if not args.flat_agent:
        agent = HierarchicalAgent(model, meta_args, sub_args, args, env.action_space.sample, subgoal_dim, subgoal_ranges, target_dim)
    else: 
        agent = Agent(model, sub_args, args, env.action_space.sample)
    set_seeds(env, args.seed)
    return env, agent 

def exponential_decay(total_steps, init_step=100, min_step=10):
        '''Gives out an exponentially decayed step size for the 
        meta-agent. step = init_step * exp(- tau * iteration)
        The decay rate tau is chosen such that the step decays to the min
        step after half the total training steps.
        :param init_step: Initial stepsize. E.g. 100
        :param min_step: Minimal step size that will be reached.
        :param decay_rate:
        '''
        decay_rate = -np.math.log(min_step / init_step) * (2 / total_steps)
        step = init_step
        while step > min_step:
            step = step * np.exp(-decay_rate)
            yield (step * np.exp(-decay_rate))
        while True:
            yield (min_step)


