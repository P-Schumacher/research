import tensorflow as tf
import numpy as np
from copy import deepcopy
from collections import namedtuple
from matplotlib import pyplot as plt
import datetime
from pudb import set_trace
from collections import deque
import agent_files.TD3_tf
from utils.replay_buffers import ReplayBuffer
from agent_files.Agent import Agent
import wandb

#wandb.init(project='research', entity='rlpractitioner')

sub_Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'goal'])
meta_Transition = namedtuple('Transition', ['state', 'goal', 'done'])

class TransitBuffer(ReplayBuffer):
    '''This class can be used like a normal ReplayBuffer of a non Hierarchical agent. It stores generated goals internally and autonomously creates
    the correct sub- and meta-agent transitions. Just don't sample directly from it.'''
    def __init__(self, agent, sub_state_dim, action_dim, meta_state_dim, subgoal_dim, target_dim, args):
        assert sub_state_dim == meta_state_dim - target_dim + subgoal_dim
        # Algo objects
        self.args = args
        self.agent = agent
        self.goal_type = args.goal_type
        self.offpolicy = args.offpolicy
        # Dimensions 
        self.subgoal_dim = subgoal_dim 
        self.target_dim = target_dim
        # Reward scales 
        self.sub_rew_scale = args.sub_rew_scale
        self.meta_rew_scale = args.meta_rew_scale
        # Replay Buffers
        self.sub_replay_buffer = ReplayBuffer(sub_state_dim, action_dim, args.c_step)
        self.meta_replay_buffer = ReplayBuffer(meta_state_dim, subgoal_dim, args.c_step)
        self.state_seq = np.ones(shape=[args.c_step, sub_state_dim - subgoal_dim + target_dim]) * np.inf 
        self.action_seq = np.ones(shape=[args.c_step, action_dim]) * np.inf 
        # Control flow variables
        self.sum_of_rewards = 0
        self.ep_rewards = 0
        self.init = False  # The buffer.add() does not create a transition in s0 as we need s1
        self.meta_time = False
        self.needs_reset = False
        self.timestep = -1
        self.ptr = 0
            
    def add(self, state, action, reward, next_state, done):
        '''Adds the transitions to the appropriate buffers. Goal is saved at
        each timestep to be used for the transition of the next timestep. At 
        an episode end, the meta agent recieves a transition of t:t_end, 
        regardless if it has been c steps.'''
        self.timestep += 1
        if self.needs_reset:
            raise Exception("You need to reset the agent if a 'done' has occurred in the environment.")
        if not self.init:  # Variables are saved in the first call, transitions constructed in later calls.
            if done:
                self.finish_one_step_episode(state, action, reward, next_state, done)
                self.needs_reset = True
            self.save_sub_transition(state, action, reward, next_state, done, self.goal)
            self.save_meta_transition(state, self.goal, done)
            self.sum_of_rewards += reward
            self.init = True
            return None
        self.finish_sub_transition(self.goal, reward)
        self.save_sub_transition(state, action, reward, next_state, done, self.goal)
        if self.meta_time:
            self.finish_meta_transition(state, done)
            self.save_meta_transition(state, self.goal, done)
            self.sum_of_rewards = 0
        if done:
            self.ep_rewards = 0
            self.agent.select_action(next_state)
            next_goal = self.goal
            self.finish_sub_transition(self.goal, reward)
            self.finish_meta_transition(next_state, done)
            self.needs_reset = True
            return self.ep_rewards

        self.sum_of_rewards += reward
    
    def collect_seq_state_actions(self, state, action):
        self.state_seq[self.ptr] = state
        self.action_seq[self.ptr] = action
        self.ptr += 1

    def finish_sub_transition(self, next_goal, reward):
        old = self.load_sub_transition()
        intr_reward = self.compute_intr_reward(old.goal, old.state, old.next_state, self.goal_type) * self.sub_rew_scale
        if self.args.ri_re:
            intr_reward += reward
        self.ep_rewards += intr_reward
        self.add_to_sub(old.state, old.goal, old.action, intr_reward, old.next_state, next_goal, old.done)

    def finish_meta_transition(self, next_state, done):
        old = self.load_meta_transition()
        self.add_to_meta(old.state, old.goal, self.sum_of_rewards * self.meta_rew_scale, next_state, done)

    def finish_one_step_episode(self, state, action, reward, next_state, done):
        goal = self.goal
        self.agent.select_action(next_state)
        next_goal = self.goal
        intr_reward = self.compute_intr_reward(goal, state, next_state, self.goal_type)
        self.add_to_sub(state, goal, action, intr_reward, next_state, next_goal, done)
        self.add_to_meta(state, goal, reward * self.meta_rew_scale, next_state, done)

    def add_to_sub(self, state, goal, action, intr_reward, next_state, next_goal, extr_done):
        '''Adds the relevant transition to the sub-agent replay buffer.'''
        self.collect_seq_state_actions(state, action)
        # Remove target from sub-agent state. Conform with paper code.
        state = state[:-self.target_dim]  
        next_state = next_state[:-self.target_dim]
        cat_state = np.concatenate([state, goal])
        cat_next_state = np.concatenate([next_state, next_goal])
        # Zero out x and y for sub-agent. Conform with paper code.
        # Makes only sense with ee_pos or torso_pos, not joints
        if self.args.zero_obs:
            cat_state[:self.args.zero_obs] = 0
            cat_next_state[:self.args.zero_obs] = 0
        self.sub_replay_buffer.add(cat_state, action, intr_reward,
                                   cat_next_state, extr_done, 0, 0)
        
    def save_sub_transition(self, state, action, reward, next_state, done, goal):
        self.sub_transition = sub_Transition(state, action, reward, next_state, done, goal)

    def load_sub_transition(self):
        return self.sub_transition

    def add_to_meta(self, state, goal, sum_of_rewards, next_state_c, done_c):
        self.meta_replay_buffer.add(state, goal, sum_of_rewards, next_state_c, done_c, self.state_seq,
                                    self.action_seq)
        self._reset_sequence()

    def _reset_sequence(self):
        if self.offpolicy:
            self.state_seq[:] = np.inf
            self.action_seq[:] = np.inf
            self.ptr = 0
    
    def save_meta_transition(self, state, goal, done):
        self.meta_transition = meta_Transition(state, goal, done)

    def load_meta_transition(self):
        return self.meta_transition

    def compute_intr_reward(self, goal, state, next_state, goal_type):
        '''Computes the intrinsic reward for the sub agent. It is the L2-norm between the goal and the next_state, restricted to those dimensions that
        the goal covers. In the HIRO Ant case: BodyPosition: x, y, z BodyOrientation: a, b, c, d JointPositions: 2 per leg. Total of 15 dims. State space
        also contains derivatives of those quantities (i.e. velocity). Total of 32 dims.'''
        dim = self.subgoal_dim
        if goal_type == "Absolute":
            rew = -tf.norm(next_state[:dim] - goal)  # complete absolute goal reward
        elif goal_type == "Direction":
            rew =  -tf.norm(state[:dim] + goal - next_state[:dim])  # complete directional reward
        else:
            raise Exception("Goal type has to be Absolute or Direction")
        return rew


class HierarchicalAgent(Agent):
    def __init__(self, model_cls, meta_args, sub_args, args, random_action_fn, subgoal_dim, subgoal_ranges, target_dim):
        # Args parameters
        self.args = args
        self.meta_mock = args.meta_mock
        self.sub_mock = args.sub_mock
        self.c_step = self.args.c_step          
        # Explicit parameters
        self.subgoal_ranges = subgoal_ranges
        self.target_dim = target_dim
        self.file_name = self._create_file_name(self.args.policy, self.args.env, self.args.seed)
        self.random_action_fn = random_action_fn
        # Policy Kwargs 
        sub_args, meta_args = self._get_sub_and_meta_args(meta_args, sub_args, subgoal_dim, subgoal_ranges, target_dim)
        self.meta_state_dim = meta_args["state_dim"]
        self.subgoal_dim = meta_args["action_dim"]
        self.sub_state_dim = sub_args["state_dim"]
        self.action_dim = sub_args["action_dim"]
        # Models and buffer
        self.sub_agent = model_cls(**sub_args)
        self.meta_agent = model_cls(**meta_args)
        self.transitbuffer = TransitBuffer(self, self.sub_state_dim, self.action_dim, self.meta_state_dim,
                                           self.subgoal_dim, self.target_dim, args)
        # Logic variables
        # Set this to its maximum, such that we query the meta agent in the first iteration
        self.goal_counter = self.c_step 
        # Counts how many evaluations have been done
        self.evals = 0  
        
    def select_action(self, state):
        '''Selects an action from the sub agent to output. For this a goal is queried from the meta agent and
        saved(!) for the add-fct. In this function, no noise is added.'''
        self._get_meta_goal(state)
        action = self._get_sub_action(state)
        return action

    def select_noisy_action(self, state):
        '''Selects an action from sub- and meta-agent and adds gaussian noise to it. Then clips actions appropriately to the sub.max_action
        or the META_RANGES'''
        # TODO scale noise to action size
        action = self.select_action(state) + self.gaussian_noise(self.args.sub_noise, self.action_dim)
        if self.meta_time:
            self.goal = self.goal + self.gaussian_noise(self.args.meta_noise, self.subgoal_dim)
            self.goal = tf.clip_by_value(self.goal, -self.subgoal_ranges, self.subgoal_ranges)
        return tf.clip_by_value(action, -self.sub_agent.max_action, self.sub_agent.max_action)
    
    def random_action(self, state):
        ''' In HIRO this fct. still returns just normal gaussian exploration, NOT uniform sampling as in TD3 paper. 
        This is in line with the HIRO implementation.
        '''
        return self.select_noisy_action(state)

    def train(self, time_step):
        '''Train both agents for as many steps as episode had.'''
        self.sub_agent.train(self.transitbuffer.sub_replay_buffer, self.args.batch_size, time_step)
        if time_step % self.c_step == 0:
            self.meta_agent.train(self.transitbuffer.meta_replay_buffer,
                                                            self.args.batch_size, time_step, self.sub_agent.actor)

    def replay_add(self, state, action, reward, next_state, done):
        return self.transitbuffer.add(state, action, reward, next_state, done)
        
    def save_model(self, string):
        '''Saves the weights of sub and meta agent to a file.'''
        self.sub_agent.actor.save_weights(string + "_sub_actor")
        self.sub_agent.critic.save_weights(string + "_sub_critic")
        self.meta_agent.actor.save_weights(string + "_meta_actor")
        self.meta_agent.critic.save_weights(string + "_meta_critic")

    def load_model(self, string):
        '''Loads the weights of sub and meta agent from a file.'''
        self.sub_agent.actor.load_weights(string + "_sub_actor")
        self.sub_agent.critic.load_weights(string + "_sub_critic")
        self.meta_agent.actor.load_weights(string + "_meta_actor")
        self.meta_agent.critic.load_weights(string + "_meta_critic")

    def reset(self):
        '''Want this reset such that the meta agent proposes the first goal every
        episode and we don't get the last goal from the previous episode. It also
        prevents lingering old transition components from being used. This also resets
        the sum of rewards for the meta agent.'''
        self.goal_counter = self.c_step
        self.transitbuffer.init = False
        self.transitbuffer.sum_of_rewards = 0
        self.transitbuffer.needs_reset = False
        self.transitbuffer.state_seq[:] = np.inf
        self.transitbuffer.action_seq[:] = np.inf
        self.transitbuffer.ptr = 0
    
    def _maybe_mock(self, goal):
        '''Replaces the subgoal by a constant goal that is put in by hand. For debugging and understanding.'''
        if not self.args.mock:
            return goal
        if self.args.ee_pos:
            mock_goal = np.zeros([6,])
            mock_goal[:3] = [0.625, -0.01, 0.58]
            return mock_goal
        np.load('end_Eff_pos.npy')
        mock_joint_goal = np.zeros([11,])
        mock_joint_goal[:7] = np.load('end_Eff_position.npy')
        return mock_joint_goal
    
    def _maybe_move_over_table(self, goal, move=False):
        if move:
            print("moved")
            goal = tf.constant([0,0,1,0,0,0], dtype=tf.float32) + goal 
        return goal 
    
    def _get_sub_and_meta_args(self, meta_args, sub_args, subgoal_dim, subgoal_ranges, target_dim):
        '''Prepares the keyword arguments for the TD3 algo for the meta and the sub agent. The meta agent outputs
        an action corresponding to the META_RANGES dimension. The state space of the sub agent is the sum of its 
        original state space and the action space of the meta agent.'''
        meta_args["action_dim"] = subgoal_ranges.shape[-1]
        meta_args["max_action"] = tf.convert_to_tensor(self.subgoal_ranges)
        sub_args["state_dim"] = sub_args["state_dim"] - target_dim + meta_args["action_dim"]
        meta_args["no_candidates"] = self.args.no_candidates
        meta_args["c_step"] = self.args.c_step
        return sub_args, meta_args
    
    def _get_meta_goal(self, state):
        # TODO correctly give meta goal with meta_time and not training
        if self.meta_mock:
            self.goal = self.meta_mock_goal
        self.goal, self.meta_time = self._sample_goal(state, self.args.goal_type)

    def _get_sub_action(self, state):
        if self.sub_mock:
            action = np.zeros([8,], dtype=np.float32)
            action[:7] = self.goal
            return action 
        # Zero out x,y for sub agent. Then take target away in select_action. Conform with paper.
        if self.args.zero_obs:
            state = state.copy()
            state[:self.args.zero_obs] = 0
        return  self.sub_agent.select_action(np.concatenate([state[:-self.target_dim], self.goal]))
    
    def _goal_transition_fn(self, goal, previous_state, state, goal_type="Absolute"):
        '''When using directional goals, we have to transition the goal at every 
        timestep to keep it at the same absolute position for n timesteps. In absolute 
        mode, this is just the identity.'''
        if goal_type == "Absolute":
            return goal
        elif goal_type == "Direction":
            dim = self.subgoal_dim
            return previous_state[:dim] + goal - state[:dim]
        else:
            raise Exception("Enter a valid type for the goal, Absolute or Direction.")
    
    def _check_inner_done(self, state, next_state, goal, goal_type):
        '''Checks if the sub-agent has managed to reach the subgoal and then calls a new subgoal.'''
        inner_done = self._inner_done_cond(state, next_state, goal, goal_type)
        if self.args.log:
            wandb.log({'distance_to_goal':inner_done}, commit=False)
    
    def _inner_done_cond(self, state, next_state, goal, goal_type):
        dim = self.subgoal_dim
        if goal_type == 'Absolute':
            diff = next_state[:dim] - goal[:dim]
        else:
            diff = state[:dim] + goal - next_state[:dim]
        return tf.norm(diff) 

    def _sample_goal(self, state, goal_type):
        '''Either output the existing goal or query the meta agent for a new goal, depending on the timestep. The 
        goal and the meta_time boolean are saved to the transitbuffer for later use by the add-fct.'''
        assert self.transitbuffer.goal_type == goal_type
        self.goal_counter += 1
        if self.goal_counter < self.c_step:
            meta_time = False
            goal = self._goal_transition_fn(self.goal, self.prev_state, state, goal_type)
            self.prev_state = state
        else:
            self.goal_counter = 0
            self.prev_state = state
            meta_time = True
            goal = self.meta_agent.select_action(state)
            goal = self._maybe_move_over_table(goal)
            goal = self._maybe_mock(goal)
        self._check_inner_done(self.prev_state, state, goal, goal_type)
        return goal, meta_time
    
    def _eval_policy(self, env, env_name, seed, time_limit, visit, eval_episodes=5):
        '''Runs policy for X episodes and returns average reward, average intrinsic reward and success rate.
        Differen seeds are used for the eval environments. visit is a boolean that decides if we record visitation
        plots.'''
        env.seed(self.args.seed + 100)
        avg_ep_reward = []
        avg_intr_reward = []
        success_rate = 0
        visitation = np.zeros((time_limit, env.observation_space.shape[0]))
        for episode_nbr in range(eval_episodes):
            print("eval number:"+str(episode_nbr)+" of "+str(eval_episodes))
            step = 0
            state, done = env.reset(evalmode=False), False
            self.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                avg_ep_reward.append(reward)
                avg_intr_reward.append(self.transitbuffer.compute_intr_reward(self.goal, state, next_state, self.args.goal_type))
                state = next_state
                if visit:
                    visitation[step, :] = state
                step += 1
                if done and step < env._max_episode_steps:
                    success_rate += 1
            if visit:
                np.save("./results/visitation/visitation_"+str(self.evals)+"_"+str(episode_nbr)+"_"+str(self.file_name), visitation)

        avg_ep_reward = np.sum(avg_ep_reward) / eval_episodes
        avg_intr_reward = np.sum(avg_intr_reward) / eval_episodes
        success_rate = success_rate / eval_episodes
        print("---------------------------------------")
        print("Evaluation over {eval_episodes} episodes: "+str(avg_ep_reward))
        print("---------------------------------------")
        self.evals += 1
        return avg_ep_reward, avg_intr_reward, success_rate

    @property
    def goal(self):
        return self.transitbuffer.goal

    @goal.setter
    def goal(self, goal):
        self.transitbuffer.goal = goal

    @property
    def meta_time(self):
        return self.transitbuffer.meta_time

    @meta_time.setter
    def meta_time(self, meta_time):
        self.transitbuffer.meta_time = meta_time

