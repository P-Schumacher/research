import tensorflow as tf
import os
import numpy as np
from agent_files.Agent import Agent
from utils.transitbuffer import TransitBuffer
import wandb
from pudb import set_trace

class HierarchicalAgent(Agent):
    def __init__(self, agent_cnf, buffer_cnf, main_cnf, env_spec, model_cls, subgoal_dim):
        self._prepare_parameters(agent_cnf, main_cnf, env_spec, subgoal_dim)
        self._self_prepare_algo_objects(agent_cnf, buffer_cnf, main_cnf, env_spec, model_cls, subgoal_dim)
        self._prepare_control_variables()
        
    def select_action(self, state, noise_bool=False):
        '''Selects an action from the sub agent to output. For this a goal is queried from the meta agent and
        saved(!) for the add-fct. Depending on input, gaussian noise is added to the goal and the action. Goal smoothing 
        means taking the Polyak Average of the old and the new goal. 
        :param state: State of the MDP.
        :param noise_bool: Boolean indicating if noise should be added to the action and the goal.
        :return action: The action that the sub-agent takes.'''
        self._get_meta_goal(state)
        if self.meta_time:
            self._maybe_apply_goal_clipnoise(noise_bool)
            # Need to correct goal after applying noise
            self._maybe_goal_smoothing()
        action = self._get_sub_action(state) 
        return self._maybe_apply_action_clipnoise(action, noise_bool)
    
    def train(self, timestep, episode_steps):
        '''Train the agent with 1 minibatch. The meta-agent is trained every c_step steps.'''
        sub_avg = np.zeros([6,], dtype=np.float32) 
        meta_avg = np.zeros([6,], dtype=np.float32) 
        for train_index in range(episode_steps):
            sub_avg = sub_avg + [self._train_sub_agent(timestep, train_index) if self._train_sub else [0 for x in sub_avg]][0]
            if (not train_index % self._c_step) and train_index:
                meta_avg = meta_avg + [self._train_meta_agent(timestep, train_index) if self._train_meta else [0 for x in meta_avg]][0]
        self._maybe_log_training_metrics(sub_avg / episode_steps, meta_avg / episode_steps, timestep)

    def replay_add(self, state, action, reward, next_state, done, success_cd):
        '''Adds a transition to the replay buffer.'''
        return self._transitbuffer.add(state, action, reward, next_state, done, success_cd)
        
    def save_model(self, string):
        '''Saves the weights of sub and meta agent to a file.'''
        self._sub_agent.actor.save_weights(string + "_sub_actor")
        self._sub_agent.critic.save_weights(string + "_sub_critic")
        self._meta_agent.actor.save_weights(string + "_meta_actor")
        self._meta_agent.critic.save_weights(string + "_meta_critic")

    def load_model(self, string):
        '''Loads the weights of sub and meta agent from a file.'''
        self._sub_agent.actor.load_weights(string + "_sub_actor")
        self._sub_agent.critic.load_weights(string + "_sub_critic")
        self._meta_agent.actor.load_weights(string + "_meta_actor")
        self._meta_agent.critic.load_weights(string + "_meta_critic")

    def reset(self):
        '''Want this reset such that the meta agent proposes the first goal every
        episode and we don't get the last goal from the previous episode. It also
        prevents lingering old transition components from being used. This also resets
        the sum of rewards for the meta agent.'''
        self._goal_counter = self._c_step
        self._transitbuffer.reset()

    def _self_prepare_algo_objects(self, agent_cnf, buffer_cnf, main_cnf, env_spec, model_cls, subgoal_dim):
        self._file_name = self._create_file_name(main_cnf.model, main_cnf.env, main_cnf.descriptor)
        
        meta_env_spec, sub_env_spec = self._build_modelspecs(env_spec)
        self._transitbuffer = TransitBuffer(self, sub_env_spec, meta_env_spec, subgoal_dim, self._target_dim, main_cnf, agent_cnf,
                                           buffer_cnf)
        self._sub_agent = model_cls(**sub_env_spec, **agent_cnf.sub_model)
        self._meta_agent = model_cls(**meta_env_spec, **agent_cnf.meta_model)

    def _prepare_control_variables(self):
        # Set * goal_counter* to its maximum, such that we query the meta agent in the first iteration
        self._goal_counter = self._c_step 
        self._orig_goal = tf.zeros([1,])
        self._evals = 0  
        self._init = False
        # we need an initial goal for goal smoothing
        if self._smooth_goal:
            self._prev_goal = np.zeros(shape=[3, ], dtype=np.float32)


    def _train_sub_agent(self, timestep, train_index):
        *metrics, = self._sub_agent.train(self.sub_replay_buffer, 
                                          self._batch_size, 
                                          timestep, 
                                          (self._log and not self._minilog))
        return metrics 

    def _train_meta_agent(self, timestep, train_index):
        *metrics, = self._meta_agent.train(self.meta_replay_buffer, 
                                           self._batch_size, 
                                           timestep, 
                                           (self._log and not self._minilog), 
                                           self._sub_agent.actor,
                                           self._sub_agent)
        return metrics 

    def _maybe_apply_action_clipnoise(self, action, noise_bool):
        '''Applies gaussian noise to the proposed action and then clips it to
        the allowed maximum and minimum ranges.'''
        if noise_bool:
            action += self._gaussian_noise(self._sub_noise, self._action_dim)
            return  tf.clip_by_value(action, -self._sub_agent._max_action, self._sub_agent._max_action)
        return action

    def _maybe_apply_goal_clipnoise(self, noise_bool=False):
        '''Applies gaussian noise to the proposed goal and then clips it to
        the allowed maximum and minimum ranges.'''
        if noise_bool:
            self.goal += self._gaussian_noise(self._meta_noise, self._subgoal_dim)
            if (not self._spherical_coord) and (not self._center_meta_goal):
                self.goal = tf.clip_by_value(self.goal, -self._subgoal_ranges, self._subgoal_ranges)

    def _maybe_log_training_metrics(self, sub_avg, meta_avg, timestep):
        '''Logs different training metrics such as: actor, critic loss, the norms and standard
        deviations of the gradients of each update.i
        :param sub_avg: The episode averaged metrics of the sub-agent.
        :param meta_avg: The episode averaged metrics of the meta-agent.'''
        if self._log and not self._minilog:
            should_log = [x for x in [np.array(sub_avg), np.array(meta_avg)] if x.any()]
            for avg, name in zip(should_log, ['sub', 'meta']):
                wandb.log({f'{name}/actor_loss': avg[0],
                           f'{name}/critic_loss': avg[1],
                           f'{name}/actor_gradnorm': avg[2],
                           f'{name}/critic_gradnorm': avg[3], 
                           f'{name}/actor_gradstd': avg[4],
                           f'{name}/critic_gradstd': avg[5]}, step = timestep)

    def _maybe_goal_smoothing(self):
        '''Changes the goal such that it's a linear interpolation between the previous goal
        and the newly proposed goal.'''
        if self._smooth_goal:
            if not self._init:
                self._init = True
                self._prev_goal = self.goal
            else:
                self._orig_goal = self.goal
                self.goal = self._smooth_factor * self._prev_goal + (1 - self._smooth_factor) * self.goal 
                self._prev_goal = self.goal

    def _maybe_mock(self):
        '''Replaces the subgoal by a constant goal that is put in by hand. For debugging and understanding.'''
        if not self._meta_mock:
            return 
        #mock_goal = np.array(np.random.uniform(-0.5, 0.5, size=[3,]), dtype=np.float32) 
        if self.meta_time:
            mock_goal = np.zeros_like(self.goal, dtype=np.float32)
            mock_goal[0] = 10.
            self.goal = tf.constant(mock_goal)
    
    def _maybe_center_goal(self):
        '''Adds a constant term to the generated subgoal such that it is forced to be 
        above the table.'''
        if self._center_meta_goal:
            self.goal = tf.constant([0.622, -0.605, 0.86], dtype=tf.float32) + self.goal 
    
    def _maybe_spherical_coord_trafo(self):
        '''If we give a meta-goal in spherical coordinates, this transforms it back to cartesian
        coordinates.
        r = self.goal[0]
        theta = self.goal[1]
        phi = self.goal[2]
        '''
        if self._spherical_coord:
            # Output of meta is goal \in [-1,1] Now bring it in [0,1] for spherical coordinates
            r = 1.867/2 - 0.1
            self.goal = (self.goal - (-1)) / (1 - (-1))
            x = self.goal[0] * r * np.sin(self.goal[1] * np.pi) * np.cos(self.goal[2] * np.pi)
            y = self.goal[0] * r * np.sin(self.goal[1] * np.pi) * np.sin(self.goal[2] * np.pi)
            z = self.goal[0] * r * np.cos(self.goal[1] * np.pi)
            x += 0.622
            y -= 0.605
            z += 0.86
            self.goal = tf.constant(np.array([x, y, z], dtype=np.float32))
    
    def _build_modelspecs(self, env_spec):
        '''Prepares the keyword arguments for the TD3 algo for the meta and the sub agent. The meta agent outputs
        an action corresponding to the META_RANGES dimension. The state space of the sub agent is the sum of its 
        original state space and the action space of the meta agent.'''
        meta_env_spec = env_spec.copy()
        sub_env_spec = env_spec.copy()
        meta_env_spec['action_dim'] = len(self._subgoal_ranges)
        meta_env_spec['max_action'] = self._subgoal_ranges
        sub_env_spec['state_dim'] = sub_env_spec['state_dim'] - self._target_dim + meta_env_spec['action_dim']
        if self._smooth_goal:
            meta_env_spec['state_dim'] = meta_env_spec['state_dim'] + self._subgoal_dim
        return meta_env_spec, sub_env_spec

    def _get_meta_goal(self, state):
        '''Queries a goal from the meta_agent and applies several transformations if enabled.'''
        self._maybe_modify_smoothed_state(state)
        self._sample_goal(self._meta_state)
        self._maybe_mock()
        self._check_inner_done(self._meta_state)
        if self.meta_time:
            self._maybe_spherical_coord_trafo()
            self._maybe_center_goal()

    def _maybe_modify_smoothed_state(self, state):
        '''Concatenates the previous given goal to the state vector for the
        meta-agent. As we average old and new goals, the meta-agent needs
        a way to be aware of past positions of the goal.'''
        if not self._smooth_goal:
            self._meta_state = state
        else:
            self._meta_state = np.concatenate([state, self._prev_goal], axis=0)

    def _get_sub_action(self, state):
        '''Gets the action from the sub-agent. Can use a mock-sub agent which
        outputs a pre-specified action. *zero_obs* determines how many elements
        of the state of the sub-agent should be zeroed out. Was used in HIRO 
        to make sub-agent learn faster.'''
        if self._sub_mock:
            action = np.zeros([8,], dtype=np.float32)
            action[:7] = self.goal
            return action 
        # Zero out x,y for sub agent. Then take target away in select_action. Conform with HIRO paper.
        if self._zero_obs:
            state = state.copy()
            state[:self._zero_obs] = 0
        return  self._sub_agent.select_action(np.concatenate([state[:-self._target_dim], self.goal]))
    
    def _goal_transition_fn(self, goal, previous_state, state):
        '''When using directional goals, we have to transition the goal at every 
        timestep to keep it at the same absolute position for n timesteps. In absolute 
        mode, this is just the identity.'''
        if self.goal_type == "Absolute" or self.goal_type == "Huber":
            return goal
        elif self.goal_type == "Direction" or self.goal_type == 'Sparse':
            dim = self._subgoal_dim
            return previous_state[:dim] + goal - state[:dim]
        else:
            raise Exception("Enter a valid type for the goal, Absolute, Direction or Sparse.")
    
    def _check_inner_done(self, next_state):
        '''Checks how close the sub-agent has gotten to the proposed subgoal and plots it.'''
        inner_done = self._inner_done_cond(self._prev_state, next_state, self.goal)
        if self._log and not self._minilog:
            wandb.log({'distance_to_goal':inner_done}, commit=False)
    
    def _inner_done_cond(self, state, next_state, goal):
        '''Checks how close the sub-agent got to the proposed goal.'''
        dim = self._subgoal_dim
        if self.goal_type == 'Absolute':
            diff = next_state[:dim] - goal[:dim]
        else:
            diff = state[:dim] + goal - next_state[:dim]
        return tf.norm(diff) 

    def _sample_goal(self, state):
        '''Either output the existing goal or query the meta agent for a new goal, depending on the timestep. The 
        goal and the meta_time boolean are saved to the transitbuffer for later use by the add-fct.'''
        self._goal_counter += 1
        if self._goal_counter < self._c_step:
            self.meta_time = False
            self.goal = self._goal_transition_fn(self.goal, self._prev_state, state)
            self._prev_state = state
        else:
            self.meta_time = True
            self._goal_counter = 0
            self._prev_state = state
            self.goal = self._meta_agent.select_action(state)
        if self._goal_every_iteration:
            self.goal = self._meta_agent.select_action(state)
    
    def _eval_policy(self, env, seed, visit):
        '''Runs policy for X episodes and returns average reward, average intrinsic reward and success rate.
        Different seeds are used for the eval environments. visit is a boolean that decides if we record visitation
        plots.'''
        #avg_q = self._average_q_value()
        env.seed(self._seed + 100)
        avg_ep_reward = []
        avg_intr_reward = []
        rate_correct_solves = 0
        success_rate = 0
        untouchable_steps = 0
        visitation = np.zeros((env.max_episode_steps, env.observation_space.shape[0]))
        for episode_nbr in range(self._num_eval_episodes):
            print(f'eval number: {episode_nbr} of {self._num_eval_episodes}')
            step = 0
            state, done = env.reset(evalmode=True), False
            self.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                avg_ep_reward.append(reward)
                avg_intr_reward.append(self._transitbuffer.compute_intr_reward(self.goal, state, next_state, action))
                state = next_state
                if env._stop_counter < 20:
                    untouchable_steps += 1
                if visit:
                    visitation[step, :] = state
                step += 1
                if done and step < env.max_episode_steps:
                    success_rate += 1
                    if env._double_buttons:
                        if env.mega_reward:
                            rate_correct_solves += 1
            if visit:
                self._create_visit_directory()
                np.save(f'./results/visitation/{self._file_name}/visitation_{self._evals}_{episode_nbr}_{self._file_name}', visitation)

        avg_ep_reward = np.sum(avg_ep_reward) / self._num_eval_episodes
        avg_intr_reward = np.sum(avg_intr_reward) / self._num_eval_episodes
        success_rate = success_rate / self._num_eval_episodes
        rate_correct_solves = rate_correct_solves / self._num_eval_episodes
        print("---------------------------------------")
        print(f'Evaluation over {self._num_eval_episodes} episodes: {avg_ep_reward}')
        print("---------------------------------------")
        self._evals += 1
        return avg_ep_reward, avg_intr_reward, success_rate, rate_correct_solves, untouchable_steps

    def _average_q_value(self):
        '''Compute the average Q value by doing multivariate Monte Carlo Integration over
        state and actions.'''
        # TODO implement
        return 0

    def _create_visit_directory(self):
        if not os.path.exists(f'./results/visitation/{self._file_name}'):
                os.makedirs(f'./results/visitation/{self._file_name}')

    def _prepare_parameters(self, agent_cnf, main_cnf, env_spec, subgoal_dim):
        '''Unpacks the parameters from the config files to state variables.'''
        # Env specs
        self._subgoal_ranges = np.array(env_spec['subgoal_ranges'], dtype=np.float32)
        self._target_dim = env_spec['target_dim']
        self._action_dim = env_spec['action_dim']
        self._subgoal_dim = subgoal_dim
        # Main cnf
        self._minilog = main_cnf.minilog
        self._batch_size = main_cnf.batch_size
        self._c_step = main_cnf.c_step
        self._seed = main_cnf.seed
        self._log = main_cnf.log
        self._gradient_steps = main_cnf.gradient_steps
        self._visit = main_cnf.visit
        # Agent cnf
        self._center_meta_goal = agent_cnf.center_meta_goal
        self._spherical_coord = agent_cnf.spherical_coord
        self._num_eval_episodes = agent_cnf.num_eval_episodes
        self._meta_mock = agent_cnf.meta_mock
        self._sub_mock = agent_cnf.sub_mock
        self._meta_noise = agent_cnf.meta_noise
        self._sub_noise = agent_cnf.sub_noise
        self._zero_obs = agent_cnf.zero_obs
        self._train_meta = agent_cnf.train_meta
        self._train_sub = agent_cnf.train_sub
        self.goal_type = agent_cnf.goal_type
        self._smooth_goal = agent_cnf.smooth_goal
        self._smooth_factor = agent_cnf.smooth_factor
        self._goal_every_iteration = agent_cnf.goal_every_iteration

    @property
    def goal(self):
        return self._transitbuffer.goal

    @goal.setter
    def goal(self, goal):
        self._transitbuffer.goal = goal

    @property
    def meta_time(self):
        return self._transitbuffer.meta_time

    @meta_time.setter
    def meta_time(self, meta_time):
        self._transitbuffer.meta_time = meta_time

    @property
    def sub_replay_buffer(self):
        return self._transitbuffer._sub_replay_buffer

    @property
    def meta_replay_buffer(self):
        return self._transitbuffer._meta_replay_buffer

    @property
    def _prev_goal(self):
        return self._transitbuffer._prev_goal

    @_prev_goal.setter
    def _prev_goal(self, prev_goal):
        self._transitbuffer._prev_goal = prev_goal

    @property
    def _meta_state(self):
        return self._transitbuffer._meta_state

    @_meta_state.setter
    def _meta_state(self, meta_state):
        self._transitbuffer._meta_state = meta_state  

    @property
    def _orig_goal(self):
        return self._transitbuffer._orig_goal

    @_orig_goal.setter
    def _orig_goal(self, orig_goal):
        self._transitbuffer._orig_goal = orig_goal  
