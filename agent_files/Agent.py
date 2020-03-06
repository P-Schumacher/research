import os 
import sys
import numpy as np
import gym
import tensorflow as tf
from pudb import set_trace
from utils.replay_buffers import ReplayBuffer
import wandb 

class Agent:
    def __init__(self, cnf, specs, model):
        self.cnf = cnf 
        self.state_dim = specs["state_dim"]
        self.action_dim = specs["action_dim"]
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, **cnf.buffer)
        self.file_name = self._create_file_name(self.cnf.main.policy, self.cnf.main.env, self.cnf.main.seed)
        self.policy = model(**specs, **cnf.agent.sub_model) 

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
        tf.random.set_seed(self.cnf.main.seed)
        env.reset()
        env.reset(hard_reset=True)
        avg_reward, avg_intr_reward, success_rate =  self._eval_policy(env, self.cnf.main.env, self.cnf.main.seed,
                                                                       self.cnf.coppeliagym.time_limit,
                                                                       self.cnf.main.visit, self.cnf.agent.num_eval_episodes)
        self.reset()
        return avg_reward, avg_intr_reward, success_rate
    
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def _eval_policy(self, env, env_name, seed, time_limit, visit, eval_episodes=5):
        '''Runs policy for X episodes and returns average reward, average intrinsic reward and success rate.
        Different seeds are used for the eval environments. visit is a boolean that decides if we record visitation
        plots.'''
        env.seed(self.cnf.main.seed + 100)
        avg_ep_reward = []
        success_rate = 0
        for episode_nbr in range(eval_episodes):
            print("eval number:"+str(episode_nbr)+" of "+str(eval_episodes))
            step = 0
            state, done = env.reset(evalmode=True), False
            self.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                avg_ep_reward.append(reward)
                state = next_state
                step += 1
                if done and step < env._max_episode_steps:
                    success_rate += 1

        avg_ep_reward = np.sum(avg_ep_reward) / eval_episodes
        success_rate = success_rate / eval_episodes
        print("---------------------------------------")
        print("Evaluation over {eval_episodes} episodes: "+str(avg_ep_reward))
        print("---------------------------------------")
        return avg_ep_reward, 0, success_rate

    def select_action(self, state):
        state = np.array(state)
        return self.policy.select_action(state) 

    def select_noisy_action(self, state):
        state = np.array(state)
        action = self.policy.select_action(state) + self.gaussian_noise(self.cnf.agent.sub_noise)
        return tf.clip_by_value(action, -self.policy.max_action, self.policy.max_action)      
    
    def gaussian_noise(self, expl_noise, dimension=1):
        return np.random.normal(0, expl_noise, dimension) 

    def random_action(self, state):
        return self.random_action_fn()
    
    def train(self, timestep):
        m_avg = np.zeros([6, ], dtype=np.float32)
        for i in range(self.cnf.main.gradient_steps):
            *metrics, = self.policy.train(self.replay_buffer, self.cnf.main.batch_size, timestep)
            m_avg += metrics 
        m_avg /= self.cnf.main.gradient_steps
        if self.cnf.main.log:
            wandb.log({f'sub/actor_loss': m_avg[0],
                       f'sub/critic_loss': m_avg[1],
                       f'sub/critic_gradmean': m_avg[2],
                       f'sub/actor_gradmean': m_avg[3], 
                       f'sub/actor_gradstd': m_avg[4],
                       f'sub/critic_gradstd': m_avg[5]}, step = timestep)

    def replay_add(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, self.cnf.agent.sub_rew_scale * reward, done, 0, 0)

    def save_model(self, string):
        self.policy.actor.save_weights(string + "_policy_actor")
        self.policy.critic.save_weights(string + "_policy_critic")

    def load_model(self, string):
        self.policy.actor.load_weights(string + "_policy_actor")
        self.policy.critic.load_weights(string + "_policy_critic")

    def reset(self):
        pass # Not necessary for simple agent
