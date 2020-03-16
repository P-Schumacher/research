import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from contextlib import contextmanager
from pudb import set_trace
from rl_algos import TD3_tf
from environments.my_env import EnvWithGoal
from agent_files.Agent import Agent
from agent_files.HIRO import HierarchicalAgent

def get_model_class(model_name):
	if model_name == 'TD3':
		return TD3_tf.TD3

def create_env(cnf):
    '''Creates an environment from either OpenAi Gym or the GoogleBrain Mujoco AntMaze
    gym environment.
    It wraps it such that it tries to reach a global target position,
    which is appended to the obs. It takes obs[:2] as x,y coordinates.
    '''
    if cnf.main.vrep:
        # Load cool robotics env
        from environments.coppeliagym import CoppeliaEnv 
        print(f'Force mode is {cnf.coppeliagym.params.force}')
        print(f'ee pos is: {cnf.coppeliagym.params.ee_pos}')
        env = CoppeliaEnv(cnf.coppeliagym)
        print(f'Target is: {env._ep_target_pos}')
        if cnf.main.render:
            env.render()
        return env
    else:
        # *show* necessary because we need to load a different xml file with spheres
        from environments.create_maze_env import create_maze_env
        env = create_maze_env(cnf.maze_env)
        return EnvWithGoal(env, cnf.env_w_goal)

def get_env_specs(env):
        ''' Get necessary dimensions from environment to instantiate model.'''
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        if env.action_space.high.shape[0] > 1:
            max_action = env.action_space.high
        else: 
            max_action = float(env.action_space.high[0])
        subgoal_ranges = env.subgoal_ranges
        subgoal_dim = env.subgoal_dim
        target_dim = env.target_dim
        print('Env Specs: ')
        print(f'state_dim: {state_dim}')
        print(f'action_dim: {action_dim}')
        print(f'max_action: {max_action}')
        print(f'time_limit: {env.max_episode_steps}')
        print(f'subgoal_dim: {subgoal_dim}')
        print(f'target_dim: {target_dim}')
        return  {'state_dim': state_dim,
                'action_dim': action_dim,
                'max_action': max_action,
                'subgoal_ranges': subgoal_ranges,
                'target_dim': target_dim}
        
def create_directories(args):
    '''Create directories to save weights and results.'''
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if args.save_model and not os.path.exists('./models'):
        os.makedirs('./models')

def set_seeds(env, seed):
    '''Set seeds to get different random numbers for every experiment. Seeds have to be set by CMD Prompt.'''
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def create_world(cnf):
    create_directories(cnf)
    env = create_env(cnf)
    model_cls = get_model_class(cnf.main.model)
    env_spec = get_env_specs(env)
    if not cnf.main.flat_agent:
        agent = HierarchicalAgent(cnf.agent, cnf.buffer, cnf.main, env_spec, model_cls, env.subgoal_dim)
    else: 
        agent = Agent(cnf.agent, cnf.buffer, cnf.main, env_spec, model_cls)
    set_seeds(env, cnf.main.seed)
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

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
