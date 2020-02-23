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

def setup_parser(parse_string):
    # TODO replace by yaml style file cf. Julius' code
    ''' Parses command prompt arguments and outputs necessary variables.'''
    print(parse_string)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise (not used with HIRO agent)
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", action="store_true")        # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--zero_obs", default=0, type=int)          # Zeros sub state x,y (state[:2]=0)
    parser.add_argument("--visit", action="store_true")             # Saves state-trajectories of all evaluation episodes
    parser.add_argument("--time_limit", default=500, type=int)      # Specifies episode limit in the environment itself, not the surrounding script
    parser.add_argument("--render", action="store_true")            # Would you like to render the environment
    parser.add_argument("--sub_noise", default=1, type=float)       # Std of Gaussian exploration noise, these replace expl_noise for HIRO 
    parser.add_argument("--sub_rew_scale", default=1, type=float)   # Reward scaling (Simple reward * a) 
    parser.add_argument("--meta_rew_scale", default=0.1, type=float)# # Reward scaling (Simple reward * a) 
    parser.add_argument("--meta_noise", default=1, type=float)      # Std of Gaussian exploration noise, these replace expl_noise for HIRO
    parser.add_argument("--c_step", default=10, type=int)     # How often is meta agent queried (is equal to when it trains at the moment)
    parser.add_argument("--goal_type", default="Absolute", type=str)# Are goals "Absolute" or "Direction" based
    parser.add_argument("--no_candidates", default=10, type=int)    # How many goals are sampled for offpolicy correction.
    parser.add_argument("--offpolicy", action="store_true")         # OffPolicy at training sampling 
    parser.add_argument("--vrep", action="store_true")              # Should coppeliasim run 
    parser.add_argument("--force", action="store_true")             # Force or Target Vel. mode 
    parser.add_argument("--ee_pos", action="store_true")            # Should we use the ee_pos or the j_pos 
    parser.add_argument("--mock", action="store_true")              # Should we use a mock goal 
    parser.add_argument("--sparse_rew", action="store_true")        # Should we use sparse rewards (i.e. -1 or 0)
    parser.add_argument("--meta_mock", action="store_true")         # Should we use a hand-crafted constant meta_goal
    parser.add_argument("--sub_mock", action="store_true")          # Should we use PID controllers as perfect pi_lo 
    parser.add_argument("--random_target", action="store_true")     # Should we randomize table target 
    parser.add_argument("--ee_j_pos", action="store_true")     # Should we randomize table target 
    parser.add_argument("--meta_actr_lr", default=0.0001, type=float)     # Should we randomize table target 
    parser.add_argument("--meta_ctr_lr", default=0.01, type=float)     # Should we randomize table target 
    parser.add_argument("--sub_actr_lr", default=0.0001, type=float)     # Should we randomize table target 
    parser.add_argument("--sub_ctr_lr", default=0.0001, type=float)     # Should we randomize table target 
    parser.add_argument("--flat_agent", action="store_true")        # Should we use a flat agent 

    args = parser.parse_args(parse_string)
    print(args)
    return args

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
        env = create_maze_env(args.env_name, show=args.render)
        return EnvWithGoal(env, args.env_name, args.time_limit, render=args.render, show=args.render, evalmode=args.evalmode)

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
        '''print("Env Specs: ")
        print(f"state_dim: {state_dim}")
        print(f"action_dim: {action_dim}")
        print(f"max_action: {max_action}")
        print(f"time_limit: {time_limit}")
        print(f"subgoal_dim: {subgoal_dim}")
        print(f"target_dim: {target_dim}")'''
        return (state_dim, action_dim, max_action, time_limit), subgoal_dim, subgoal_ranges, target_dim

def setup(parse_string=[]):
    # Setup the training
    args = setup_parser(parse_string)
    create_directories(args)
    return args

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


