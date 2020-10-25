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
    '''Retrieves an RL agent by name.'''
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
        print(f'Target is: {env._pos_b1}')
        if cnf.main.render:
            env.render()
        return env
    else:
        # *show* necessary because we need to load a different xml file with spheres
        from environments.create_maze_env import create_maze_env
        env = create_maze_env(**cnf.maze_env)
        return EnvWithGoal(env, **cnf.env_w_goal)

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
        return  {'state_dim': state_dim,
                'action_dim': action_dim,
                'max_action': max_action,
                'subgoal_ranges': subgoal_ranges,
                'target_dim': target_dim}
        
def create_directories(args):
    '''Create directories to save weights.'''
    if args.save_model and not os.path.exists('./models'):
        os.makedirs('./experiments/models')

def set_seeds(env, seed, no_seed=False):
    '''Set seeds to get different random numbers for every experiment. Interestingly, this also resets
    accumulated but unused tensorflow memory and frees the RAM'''
    if not no_seed:
        env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

def create_world(cnf):
    assert_sanity_check(cnf)
    create_directories(cnf)
    if not cnf.main.simple_env:
        env = create_env(cnf)
    else:
        import gym
        env = gym.make('Pendulum-v0')
        env.subgoal_ranges = [1,1]
        env.subgoal_dim = 2
        env.target_dim = 0
    agent = create_agent(cnf, env)
    set_seeds(env, cnf.main.seed, no_seed=cnf.main.no_seed)
    return env, agent 

def create_agent(cnf, env):
    model_cls = get_model_class(cnf.main.model)
    env_spec = get_env_specs(env)
    if not cnf.main.flat_agent:
        agent = HierarchicalAgent(cnf.agent, cnf.buffer, cnf.main, env_spec, model_cls, env.subgoal_dim)
    else: 
        agent = Agent(cnf.agent, cnf.buffer, cnf.main, env_spec, model_cls)
    return agent

def assert_sanity_check(cnf):
    ''' Stops the program if unlogical settings were made. E.g. We want to center the ee goal around the robot but we only have a J subgoal.'''
    if cnf.main.env == 'Vrep':
        if cnf.agent.spherical_coord:
            assert cnf.coppeliagym.params.ee_pos or cnf.coppeliagym.params.ee_j_pos
            assert not cnf.agent.center_meta_goal

        if cnf.agent.center_goal:
            assert cnf.coppeliagym.params.ee_pos or cnf.coppeliagym.params.ee_j_pos
        # tf.function decorated functions cannot have conditional branch dependent output type
        for model in [cnf.agent.sub_model, cnf.agent.meta_model]:
            assert type(model.clip_cr) != int
            assert type(model.clip_ac) != int
        # For flat agents this reward is computed in the env. for HIRO it has to be computed in the transitbuffer
        assert not (cnf.coppeliagym.params.action_regularizer and cnf.agent.agent_action_regularizer)
        assert not ((not cnf.main.flat_agent) and (cnf.coppeliagym.params.action_regularizer))
        assert not (cnf.agent.center_metagoal and (cnf.agent.goal_type == 'Direction'))
        #assert not (cnf.agent.add_multiple_dones and not cnf.coppeliagym.params.sparse_rew)
        #assert not (cnf.agent.add_multiple_dones and (cnf.agent.per == 3 or cnf.agent.per == 4))
        assert isinstance(cnf.agent.per, int)  
        assert not (cnf.main.flat_agent and cnf.agent.per == 2)
        if cnf.coppeliagym.params.double_buttons:
            assert cnf.coppeliagym.sim.scene_file == 'coppelia_scenes/kuka_double.ttt' 

# RANDOM STUFF -------------------------------------------------------------------
def exponential_decay(total_steps, init_step=100, min_step=10):
        '''Gives out an exponentially decayed step size for the 
        meta-agent. step = init_step * exp(- tau * iteration)
        The decay rate tau is chosen such that the step decays to the min
        step after half the total training steps.
        :param init_step: Initial stepsize. E.g. 100
        :param min_step: Minimal step size that will be reached.
        :param decay_rate:
        :yield: The decayed c_step value.
        '''
        decay_rate = -np.math.log(min_step / init_step) * (2 / total_steps)
        step = init_step
        while step > min_step:
            step = step * np.exp(-decay_rate)
            yield (step * np.exp(-decay_rate))
        while True:
            yield (min_step)

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
