from pudb import set_trace
import tensorflow as tf
import numpy as np
from collections import namedtuple
from utils.replay_buffers import ReplayBuffer, PriorityBuffer
from utils.math_fns import huber, euclid

sub_Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'goal'])
meta_Transition = namedtuple('Transition', ['state', 'goal', 'done'])

class TransitBuffer(ReplayBuffer):
    '''This class handles adding the right sub- and meta-agent transitions
    to the respective buffers. As the goal computed for the next_state 
    is not known at the end of an episode step, we wait an additional 
    iteration before adding a transitions. This prevents the meta-agent
    from having to compute the same goal twice.
    Terminal transitions are handled by distinguishing *success_cd* and *done*.
    *done* is used to appropriately handle the end-of-episode construction of
    transitions that span multiple timesteps. *success_cd* handles the RL 
    rule of setting the expected future discounted reward of a state-action
    pair to zero.'''
    def __init__(self, agent, sub_env_spec, meta_env_spec, subgoal_dim, target_dim, main_cnf, agent_cnf, buffer_cnf):
        self._prepare_parameters(main_cnf, agent_cnf, target_dim, subgoal_dim)
        self._prepare_buffers(buffer_cnf, sub_env_spec['state_dim'], meta_env_spec['state_dim'],
                              sub_env_spec['action_dim'])
        self._prepare_control_variables()
        self._prepare_offpolicy_datastructures(sub_env_spec)
        self._agent = agent
            
    def add(self, state, action, reward, next_state, done, success_cd, FM=None):
        '''Adds the transitions to the appropriate buffers. Goal is saved at
        each timestep to be used for the transition of the next timestep. At 
        an episode end, the meta agent recieves a transition of t:t_end, 
        regardless if it has been c steps.
        :return: The intrinsic undiscounted return for the sub-agent'''
        self._timestep += 1
        if self._needs_reset:
            raise Exception("You need to reset the agent if a 'done' has occurred in the environment.")

        if not self._init:  # Variables are saved in the first call, transitions constructed in later calls.
            self._initialize_buffer(state, action, reward, next_state, done) 
        else: 
            return self._add(state, action, reward, next_state, done, success_cd, FM)

    def compute_intr_reward(self, goal, state, next_state, action):
        '''Computes the intrinsic reward for the sub agent. It is the L2-norm between the goal and the next_state, restricted to those dimensions that
        the goal covers. In the HIRO Ant case: BodyPosition: x, y, z BodyOrientation: a, b, c, d JointPositions: 2 per leg. Total of 15 dims. State space
        also contains derivatives of those quantities (i.e. velocity). Total of 32 dims.
        For robotic envs, a metabolic cost negative reward proved to be useful. It has to be incorporated in the
        sub-agent reward, not the external meta-agent reward.
        The function supports Aboslute and Relative goal distances as well as the Huber distance.'''
        dim = self._subgoal_dim
        if self.goal_type == "Absolute":
            rew = - euclid(next_state[:dim] - goal)  
        elif self.goal_type == "Direction":
            rew =  - euclid(state[:dim] + goal - next_state[:dim])  
        elif self.goal_type == "Huber":
            rew =  - huber(state[:dim] - next_state[:dim], 1.)  
        elif self.goal_type == "Sparse":
            if euclid(state[:dim] + goal - next_state[:dim]) < 0.1: 
                rew = 0
            else:
                rew = -1
        else:
            raise Exception("Goal type has to be Absolute or Direction")
        return rew - self._action_reg * tf.square(tf.norm(action))
    
    def reset(self):
        '''Resets the class for the next episode.'''
        self._init = False
        self._sum_of_rewards = 0
        self._needs_reset = False
        self._state_seq[:] = np.inf
        self._action_seq[:] = np.inf
        self._ptr = 0

    def _initialize_buffer(self, state, action, reward, next_state, done):
        '''Should be the first function called when add transitions is added after a reset() was called. 
        This function handles episodes which only last a single step. If the episode is longer, the MDP
        information is stored to create a transition in the next add. As such, transition t=0 is created
        and added to the buffer at t=1.
        :param state: Array containing the state information of the MDP. (1D atm)
        :param action: Array containing the proposed action of the MDP
        :param reward: Float containing the reward for taking action a in state s 
        :param next_state: Array containing the next_state of the MDP
        :param done: Bool indicating whether an episode has ended.
        :return: None'''
        if done:
            self._finish_one_step_episode(state, action, reward, next_state, done)
            self._needs_reset = True
        else:
            self._save_sub_transition(state, action, reward, next_state, done, self.goal)
            if np.any(self._orig_goal):
                self.goal = self._orig_goal
            self._save_meta_transition(self._meta_state, self.goal, done)
            self._sum_of_rewards += reward
            self._init = True

    def _add(self, state, action, reward, next_state, done, success_cd, FM):
        '''This function handles all of the logic for adding transitions.
        It is the first function that should be modified, all others
        are primitives.'''
        self._finish_sub_transition(self.goal, reward)
        self._save_sub_transition(state, action, reward, next_state, success_cd, self.goal)
        if self.meta_time:
            self._finish_meta_transition(self._meta_state, success_cd, FM)
            if np.any(self._orig_goal):
                self.goal = self._orig_goal
            self._save_meta_transition(self._meta_state, self.goal, success_cd)
            self._sum_of_rewards = 0
        if done:
            self._sum_of_rewards += reward
            # This implicitly computes the next goal in the transitbuffer.
            self._agent.select_action(next_state, 0) 
            self._finish_sub_transition(self.goal, reward)
            self._finish_meta_transition(self._meta_state, success_cd, FM)
            self._needs_reset = True
            intr_return = self._ep_rewards
            self._ep_rewards = 0
            return intr_return
        self._sum_of_rewards += reward

    def _collect_seq_state_actions(self, state, action):
        '''The states and actions of the sub-agent  are collected so that the offpolicy correction for the meta-agent can
        be calculated in a more efficient way.'''
        if self._offpolicy:
            self._state_seq[self._ptr] = state 
            self._action_seq[self._ptr] = action 
            self._ptr += 1 
            
    def _finish_sub_transition(self, next_goal, reward): 
        '''Completes a transition for the sub-agent and adds it to the buffer. 
        *ri_re* determines
        if the sub-agent should recieve the sum of extrinsic and intrinsic rewards.'''
        old = self._load_sub_transition() 
        intr_reward = self.compute_intr_reward(old.goal, old.state, old.next_state, old.action) * self._sub_rew_scale 
        if self._ri_re: 
            intr_reward += 0.2 * reward
        self._ep_rewards += intr_reward
        self._add_to_sub(old.state, old.goal, old.action, intr_reward, old.next_state, next_goal, old.done)

    def _finish_meta_transition(self, next_state, done, FM):
        old = self._load_meta_transition()
        self._add_to_meta(old.state, old.goal, self._sum_of_rewards * self._meta_rew_scale, next_state, done, FM)

    def _finish_one_step_episode(self, state, action, reward, next_state, done):
        '''1 step episodes are handled separately because the adding of states to 
        the replay buffers is always one step behind the environment timestep.'''
        meta_state = self._meta_state 
        orig_goal = self._orig_goal
        goal = self.goal 
        self._agent.select_action(next_state, 0)
        intr_reward = self.compute_intr_reward(goal, state, next_state, action)
        self._add_to_sub(state, goal, action, intr_reward, next_state, self.goal, done)
        self._add_to_meta(meta_state, orig_goal, reward * self._meta_rew_scale, self._meta_state, done)

    def _save_sub_transition(self, state, action, reward, next_state, done, goal):
        self.sub_transition = sub_Transition(state, action, reward, next_state, done, goal)

    def _load_sub_transition(self):
        return self.sub_transition

    def _add_to_sub(self, state, goal, action, intr_reward, next_state, next_goal, extr_done):
        '''Adds the relevant transition to the sub-agent replay buffer.
        This is the primitive for adding transitions to the sub-agent. It is
        the last function that is called.'''
        if self._offpolicy:
            self._collect_seq_state_actions(state, action)
        # Remove target from sub-agent state. Conform with paper code.
        state = state[:-self._target_dim]  
        next_state = next_state[:-self._target_dim]
        # Concatenate for the network
        cat_state = np.concatenate([state, goal])
        cat_next_state = np.concatenate([next_state, next_goal])
        # Zero out x and y for sub-agent. Conform with paper code.
        # Makes only sense with ee_pos or torso_pos, not joints
        if self._zero_obs:
            cat_state[:self._zero_obs] = 0
            cat_next_state[:self._zero_obs] = 0
            #cat_state[10:13] = 0
            #cat_next_state[10:13] = 0
        self._sub_replay_buffer.add(cat_state, action, intr_reward,
                                   cat_next_state, extr_done, 0, 0)
        
    def _add_to_meta(self, state, goal, sum_of_rewards, next_state, done, FM=None):
        '''Adds transitions to the replay buffer of the meta agent. *self._state_seq* and 
        *self._action_seq* are collected sub-agent experience transitions that are used
        to compute the offpolicy-correction.
        This is the primitive for adding transitions to the meta-agent. It is
        the last function that is called.'''
        if self._zero_meta_index:
            state[10:19] = 0.
            next_state[10:19] = 0.
        self._meta_replay_buffer.add(state, goal, sum_of_rewards, next_state, done, self._state_seq,
                                    self._action_seq)
        if sum_of_rewards != (-1 * self._meta_rew_scale * self._c_step) and self._add_multiple_dones:
            # Adding those transitions multiple times can help in sparse tasks.
            for _ in range(4):
                self._meta_replay_buffer.add(state, goal, sum_of_rewards, next_state, done, self._state_seq,
                                            self._action_seq)
        self._reset_sequence()
        #FM.train(state, next_state, sum_of_rewards)

    def _reset_sequence(self):
        '''After the sequence has been added to the meta replaybuffer, we overwrite the arrays with np.inf,
        those are handled in the offpolicy correction correctly. This enables us to have variable length 
        state and action sequences.'''
        if self._offpolicy:
            self._state_seq[:] = np.inf
            self._action_seq[:] = np.inf
            self._ptr = 0
    
    def _save_meta_transition(self, state, goal, done):
        self.meta_transition = meta_Transition(state, goal, done)

    def _load_meta_transition(self):
        return self.meta_transition

    def _prepare_offpolicy_datastructures(self, sub_env_spec):
        # We dont want to create these arrays for large c_step if we do not correct
        c_step = [1 if not self._offpolicy else self._c_step][0]
        self._state_seq = np.ones(shape=[c_step, sub_env_spec['state_dim'] - self._subgoal_dim + self._target_dim]) * np.inf 
        self._action_seq = np.ones(shape=[c_step, sub_env_spec['action_dim']]) * np.inf 

    def _prepare_control_variables(self):
        self._sum_of_rewards = 0
        self._ep_rewards = 0
        self._init = False  # The buffer.add() does not create a transition in s0 as we need s1
        self.meta_time = False
        self._needs_reset = False
        self._timestep = -1
        self._ptr = 0

    def _prepare_buffers(self, buffer_cnf, sub_state_dim, meta_state_dim, action_dim):
        '''Create simple replay buffers for higher and lower level agent separately.
        Can also use prioritized experience replay.'''
        #assert sub_state_dim == meta_state_dim - self._target_dim + self._subgoal_dim
        if not self._sub_per:
            self._sub_replay_buffer = ReplayBuffer(sub_state_dim, action_dim, buffer_cnf)
        else:
            self._sub_replay_buffer = PriorityBuffer(sub_state_dim, action_dim, buffer_cnf)

        if not self._per:
            self._meta_replay_buffer = ReplayBuffer(meta_state_dim, self._subgoal_dim, buffer_cnf)
        else:
            self._meta_replay_buffer = PriorityBuffer(meta_state_dim, self._subgoal_dim, buffer_cnf)

    def _prepare_parameters(self, main_cnf, agent_cnf, target_dim, subgoal_dim):
        '''Unpacks the cnf settings to state variables.'''
        # Env specs
        self._offpolicy = main_cnf.offpolicy
        self._target_dim = target_dim
        self._subgoal_dim = subgoal_dim
        # Main cnf
        self._c_step = main_cnf.c_step
        self._zero_meta_index = int(main_cnf.zero_meta_index)
        # Agent cnf
        self._zero_obs = agent_cnf.zero_obs
        self.goal_type = agent_cnf.goal_type
        self._smooth_goal = agent_cnf.smooth_goal
        self._sub_rew_scale = agent_cnf.sub_rew_scale
        self._meta_rew_scale = agent_cnf.meta_rew_scale
        self._ri_re = agent_cnf.ri_re
        self._action_reg = agent_cnf.agent_action_regularizer
        self._add_multiple_dones = agent_cnf.add_multiple_dones
        self._per = agent_cnf.per
        self._sub_per = agent_cnf.sub_per
