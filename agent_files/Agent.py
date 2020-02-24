import os 
import sys
import numpy as np
import gym
import tensorflow as tf
from pudb import set_trace
from utils.replay_buffers import ReplayBuffer

class Agent:
    def __init__(self, model, policy_args, args, random_action_fn):
        self.random_action_fn = random_action_fn
        self.args = args
        self.state_dim = policy_args["state_dim"]
        self.action_dim = policy_args["action_dim"]
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.file_name = self._create_file_name(self.args.policy, self.args.env, self.args.seed)
        self.policy = model(**policy_args) 

    def _create_file_name(self, policy, env, seed):
        '''Create file_name from experiment information to save model weights.'''
        file_name = str(policy)+"_"+str(env)+"_"+str(seed)
        print("---------------------------------------")
        print("Policy: "+str(policy)+", Env: "+str(env)+", Seed: "+str(seed))
        print("---------------------------------------")
        return file_name
    
    def load_model(self, policy, file_name, load_model):
        policy_file = file_name if load_model == "default" else load_model
        policy.load("./models/"+str(policy_file))
        
    def evaluation(self, env):
        '''Play N evaluation episodes where noise is turned off. We also evaluate only the [0,16] target, not a uniformly
        sampled one. The function then returns the avg reward, intrinsic reward and the success rate over the N episodes.'''
        # Set seed to clear tensorflow cache which prevents OutOfMemory error... I hate tensorflow
        tf.random.set_seed(self.args.seed)
        env.reset(hard_reset=True)
        avg_reward, avg_intr_reward, success_rate =  self._eval_policy(env, self.args.env, self.args.seed, self.args.time_limit, self.args.visit)
        self.reset()
        return avg_reward, avg_intr_reward, success_rate
    
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def _eval_policy(self, eval_env, env_name, seed, time_limit, visit, eval_episodes=5):
        eval_env.seed(self.args.seed + 100)
        avg_reward = []
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = eval_env.step(action)
                avg_reward.append(reward)
        avg = np.sum(avg_reward) / eval_episodes
        
        print("---------------------------------------")
        print("Evaluation over "+str(eval_episodes)+" episodes: "+str(avg))
        print("---------------------------------------")
        return avg, 0, 0

    def select_action(self, state):
        state = np.array(state)
        return self.policy.select_action(state) 

    def select_noisy_action(self, state):
        state = np.array(state)
        action = self.policy.select_action(state) + self.gaussian_noise(self.args.sub_noise)
        return tf.clip_by_value(action, -self.policy.max_action, self.policy.max_action)      
    def gaussian_noise(self, expl_noise, dimension=1):
        return np.random.normal(0, expl_noise, dimension) 

    def random_action(self, state):
        return self.random_action_fn()
    
    def train(self, timestep):
        self.policy.train(self.replay_buffer, self.args.batch_size, timestep)

    def replay_add(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done, 0, 0)

    def save_model(self, string):
        self.policy.actor.save_weights(string + "_policy_actor")
        self.policy.critic.save_weights(string + "_policy_critic")

    def load_model(self, string):
        self.policy.actor.load_weights(string + "_policy_actor")
        self.policy.critic.load_weights(string + "_policy_critic")

    def reset(self):
        pass # Not necessary for simple agent
