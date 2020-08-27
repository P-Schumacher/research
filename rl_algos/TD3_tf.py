import tensorflow as tf
import numpy as np
import wandb
from pudb import set_trace
from utils.math_fns import euclid, get_norm, clip_by_global_norm
from rl_algos.networks import Actor, Critic
from rl_algos.offpol_correction import off_policy_correction
from utils.math_fns import euclid

class TD3(object):
    def __init__(self, **kwargs):
        self._prepare_parameters(kwargs)
        self._prepare_algo_objects()
        self._create_persistent_tf_variables()

    def select_action(self, state):
        state = tf.convert_to_tensor(state.reshape(1, -1))
        return tf.reshape(self.actor(state), [-1])

    def update_target_models_hard(self):
        '''Copy network weights to target network weights.'''
        self.transfer_weights(self.actor, self.actor_target, tau=1)
        self.transfer_weights(self.critic, self.critic_target, tau=1)

    def transfer_weights(self, model, target_model, tau):
        ''' Transfer model weights to target model with a factor of Tau, using Polyak Averaging'''
        # Keras fct. should not be used inside tf.function graph
        W, target_W = model.weights, target_model.weights
        for idx in range(len(W)) :
            target_W[idx] = tf.math.scalar_mul(tau, W[idx]) + tf.math.scalar_mul((1 - tau), target_W[idx])
            target_model.weights[idx].assign(target_W[idx])
     
    def train(self, replay_buffer, batch_size, t, log=False, sub_actor=None, sub_agent=None, FM=None):
        state, action, reward, next_state, done, state_seq, action_seq, _ = replay_buffer.sample(batch_size)
        reward_new = self._maybe_FM_reward(state, next_state, reward, FM, log)
        action = self._maybe_offpol_correction(sub_actor, action, state, next_state, state_seq, action_seq)
        td_error = self._train_critic(state, action, reward_new, next_state, done, log, replay_buffer.is_weight)
        if self._per:
            self._prioritized_experience_update(self._per, td_error, next_state, action, reward_new, replay_buffer)
        #state, action, reward, next_state, done, state_seq, action_seq = replay_buffer.sample_low(batch_size)
        self._train_actor(state, action, reward_new, next_state, done, log, replay_buffer.is_weight)
        #td_error = self._compute_td_error(state, action, reward, next_state, done)
        #self._prioritized_experience_update(self._per, td_error, next_state, action, reward, replay_buffer)
        self.total_it.assign_add(1)
        if log:
            wandb.log({f'{self._name}/mean_weights_actor': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.actor.weights])}, commit=False)
            wandb.log({f'{self._name}/mean_weights_critic': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.critic.weights])}, commit=False)
        return self.actor_loss.numpy(), self.critic_loss.numpy(), self.ac_gr_norm.numpy(), self.cr_gr_norm.numpy(), self.ac_gr_std.numpy(), self.cr_gr_std.numpy()

    def full_reset(self):
        '''Completely resets actor and critic network to the predefined 
        reset networks.'''
        self.transfer_weights(self.actor_reset_net, self.actor, tau=1)
        self.transfer_weights(self.actor_reset_net, self.actor_target, tau=1)
        self.transfer_weights(self.critic_reset_net, self.critic, tau=1)
        self.transfer_weights(self.critic_reset_net, self.critic_target, tau=1)

    @tf.function
    def _compute_td_error(self, state, action, reward, next_state, done):
        '''Combines the current critic output and the learning target to yield
        the td_error which is differentiated to yield the gradients for the critic'''
        current_Q1, target_Q = self.get_current_estimate_and_learning_target(state, action, reward, next_state, done)
        return tf.abs(current_Q1 - target_Q)

    @tf.function
    def get_current_estimate_and_learning_target(self, state, action, reward, next_state, done):
        '''Computes the critic estimate for the given transition tuple and then computes 
        the learning target using the deterministic TD3 recursion eqn.
        Those metrics are also important diagnostic tools.'''
        if type(reward) == float:
            state = tf.reshape(state, [1, state.shape[-1]]) 
            action = tf.reshape(action, [1, action.shape[-1]]) 
            next_state = tf.reshape(next_state, [1, state.shape[-1]]) 
            reward = tf.constant(reward, shape=[1,1], dtype=tf.float32)
            done = tf.constant(done, shape=[1,1], dtype=tf.float32)
        state_action = tf.concat([state, action], 1) # necessary because keras needs there to be 1 input arg to be able to build the model from shapes
        done = tf.reshape(done, [done.shape[0], 1])
        reward = tf.reshape(reward, [reward.shape[0], 1])
        noise = tf.random.normal(action.shape, stddev=self._policy_noise)
        # this clip keeps the noisy action close to the original action
        noise = tf.clip_by_value(noise, -self._noise_clip, self._noise_clip)
        next_action = self.actor_target(next_state) + noise
        # this clip assures that we don't take impossible actions a > max_action
        next_action = tf.clip_by_value(next_action, -self._max_action, self._max_action)
        # Compute the target Q value
        next_state_next_action = tf.concat([next_state, next_action], 1)
        target_Q1, target_Q2 = self.critic_target(next_state_next_action)
        target_Q = tf.math.minimum(target_Q1, target_Q2)
        target_Q = reward + (1. - done) * self._discount ** self._nstep * target_Q
        # Critic Update
        current_Q1, _ = self.critic(state_action)
        return current_Q1, target_Q

    @tf.function
    def _train_critic(self, state, action, reward, next_state, done, log, is_weight):
        '''Training function. We assign actor and critic losses to state objects so that they can be easier recorded 
        without interfering with tf.function. I set Q terminal to 0 regardless if the episode ended because of a success cdt. or 
        a time limit. The norm and std of the updated gradients, as well as the losses are assigned to state objects of the class. 
        This is done as tf.function decorated functions are converted to static graphs and cannot handle variable return objects.
        :param : These should be explained by any Reinforcement Learning book.
        :return: None'''
        state_action = tf.concat([state, action], 1) # necessary because keras needs there to be 1 input arg to be able to build the model from shapes
        done = tf.reshape(done, [done.shape[0], 1])
        reward = tf.reshape(reward, [reward.shape[0], 1])
        noise = tf.random.normal(action.shape, stddev=self._policy_noise)
        # this clip keeps the noisy action close to the original action
        noise = tf.clip_by_value(noise, -self._noise_clip, self._noise_clip)
        next_action = self.actor_target(next_state) + noise
        # this clip assures that we don't take impossible actions a > max_action
        next_action = tf.clip_by_value(next_action, -self._max_action, self._max_action)
        # Compute the target Q value
        next_state_next_action = tf.concat([next_state, next_action], 1)
        target_Q1, target_Q2 = self.critic_target(next_state_next_action)
        target_Q = tf.math.minimum(target_Q1, target_Q2)
        target_Q = reward + (1. - done) * self._discount ** self._nstep * target_Q
        # Critic Update
        with tf.GradientTape() as tape:
            current_Q1, current_Q2 = self.critic(state_action)
            critic_loss = (self.critic_loss_fn(current_Q1, target_Q) 
                        + self.critic_loss_fn(current_Q2, target_Q))
            # 6 because Q losses + L2-regul losses
            assert len(self.critic.losses) == 6
            # critic.losses gives us the regularization losses from the layers
            critic_loss += sum(self.critic.losses)

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        gradients, norm = clip_by_global_norm(gradients, self._clip_cr)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        self._maybe_log_critic(gradients, norm, critic_loss, log)
        return tf.abs(target_Q - current_Q1)
    
    @tf.function
    def _train_actor(self, state, action, reward_new, next_state, done, log, is_weight):
        # Can't use *if not* in tf.function graph
        if self.total_it % self._policy_freq == 0:
            # Actor update
            with tf.GradientTape(persistent=False) as tape:
                action = self.actor(state)
                state_action = tf.concat([state, action], 1)
                actor_loss = self.critic.Q1(state_action)
                mean_actor_loss = -tf.math.reduce_mean(actor_loss)
            gradients = tape.gradient(mean_actor_loss, self.actor.trainable_variables)
            gradients, norm  = clip_by_global_norm(gradients, self._clip_ac)
            self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
            self.transfer_weights(self.actor, self.actor_target, self._tau)
            self.transfer_weights(self.critic, self.critic_target, self._tau)
            self._maybe_log_actor(gradients, norm, mean_actor_loss, log) 

    def _maybe_goal_regul(self, action, reward, next_state, state_seq, action_seq, sub_agent):
        if self._name == 'meta' and (self._goal_regul or self._distance_goal_regul):
            return self._goal_regularization(action, reward, next_state, state_seq, action_seq, sub_agent)
        else:
            return reward

    def _maybe_offpol_correction(self, sub_actor, action, state, next_state, state_seq, action_seq):
        if self._name == 'meta' and self._offpolicy:
            return off_policy_correction(self._subgoal_ranges, self._target_dim, sub_actor, action, state, next_state,
                                         self._no_candidates, self._c_step, state_seq, action_seq, self._zero_obs) 
        else:
            return action

    def _maybe_FM_reward(self, state, next_state, reward,  FM, log):
        '''Uses a learned ForwardModel (or reward model) to
        replace the reward during learning'''
        if self._use_FM and self.total_it >= 30000:
            reward_FM = FM.forward_pass(state, next_state, reshape=False)
            #reward_FM *= 0.1
            high_rews = tf.where(reward != -1.)[:,0]
            low_rews = tf.where(reward == -1.)[:,0]
            high_preds = tf.gather(reward_FM, high_rews)
            low_preds = tf.gather(reward_FM, low_rews)
            high_rews = tf.gather(reward, high_rews)
            low_rews = tf.gather(reward, low_rews)
            high_log = tf.reduce_sum(tf.abs(high_preds - high_rews))
            low_log = tf.reduce_sum(tf.abs(low_preds - low_rews))
            if log:
                wandb.log({'FM/agentbatchRerror_high': high_log, 'FM/agentbatchRerror_low': low_log,
                           'FM/agentbatch_lowpred': tf.reduce_mean(low_preds), 'FM/agentbatch_highpred':
                           tf.reduce_mean(high_preds), 'FM/agentbatch_lowrew': tf.reduce_mean(low_rews),
                           'FM/agentbatch_highrew':
                           tf.reduce_mean(high_rews)}, commit=False)
            return reward_FM 
        else:
            return reward

    def _goal_regularization(self, action, reward, next_state, state_seq, action_seq, sub_agent):
        #errors = []
        #for idx, x in enumerate(state_seq):
        #    y = action_seq[idx]
        #    to_append = self._get_error(x, y, action[idx], reward[idx], sub_agent)
        #    errors.append(to_append)
        #errors = tf.reshape(errors, [len(errors), 1])
        #return reward + self.goal_regul * tf.abs(errors) - tf.reshape(self.distance_goal_regul *  euclid(next_state[:, :action.shape[1]] - action, axis=1), [128,1])
        return reward - tf.reshape(self._distance_goal_regul *  euclid(next_state[:, :action.shape[1]] - action, axis=1), [128,1])

    def _goal_transit_fn(self, goal, state, next_state):
        dim = goal.shape[0]
        return state[:dim] + goal - state[:dim]

    def _get_error(self, state_stack, action_stack, goal, sumrew, sub_agent):
        goal = tf.reshape(goal, [1, goal.shape[0]])
        sum_of_td_errors = tf.constant(0.0, shape=[1,1])
        for i in range(state_stack.shape[0] - 1):
            if tf.reduce_any(tf.math.is_inf(state_stack[i+1])):
                break
            state = state_stack[i,:-self._target_dim]
            state = tf.reshape(state, [1, state.shape[0]])
            state = tf.concat([state, goal], axis=1)
            low_action = action_stack[i,:]
            low_action = tf.reshape(low_action, [1, low_action.shape[0]])
            goal = self._goal_transit_fn(goal, state_stack[i], state_stack[i+1])
            next_state = state_stack[i+1,:-self._target_dim]
            next_state = tf.reshape(next_state, [1, next_state.shape[0]])
            next_state = tf.concat([next_state, goal], axis=1)
            intr_reward = self._compute_intr_rew(goal, state_stack[i], state_stack[i+1])  
            #low_action = sub_agent.actor(state)
            #low_action = low_action + tf.random.normal(shape=low_action.shape, mean=0, stddev=1.4)
            error = self._compute_td_error_copy(sub_agent, state, low_action, intr_reward, next_state)
            sum_of_td_errors += error
        return sum_of_td_errors

    def _compute_intr_rew(self, goal, state, next_state):
        state = tf.reshape(state, [1, state.shape[0]])
        next_state = tf.reshape(next_state, [1, next_state.shape[0]])
        dim = goal.shape[1]
        rew = -euclid(state[:, :dim] + goal - next_state[:, :dim], axis=1)
        #rew = -euclid(goal - next_state[:, :dim], axis=1)
        return rew

    @tf.function
    def _compute_td_error_copy(self, sub_agent, state, action, reward, next_state):
        # ATTENTION HAVE REMOVED TERMINAL STATE CONDITION FOR THIS CASE
        state_action = tf.concat([state, action], 1) # necessary because keras needs there to be 1 input arg to be able to build the model from shapes
        reward = tf.reshape(reward, [1, 1])
        noise = tf.random.normal(action.shape, stddev=self._policy_noise)
        # this clip keeps the noisy action close to the original action
        noise = tf.clip_by_value(noise, -sub_agent.noise_clip, sub_agent.noise_clip)
        next_action = sub_agent.actor_target(next_state) + noise
        # this clip assures that we don't take impossible actions a > max_action
        next_action = tf.clip_by_value(next_action, -sub_agent._max_action, sub_agent._max_action)
        # Compute the target Q value
        next_state_next_action = tf.concat([next_state, next_action], 1)
        target_Q1, target_Q2 = sub_agent.critic_target(next_state_next_action)
        target_Q = tf.math.minimum(target_Q1, target_Q2)
        target_Q = reward + self._discount * target_Q
        # Critic Update
        current_Q1, current_Q2 = sub_agent.critic(state_action)
        return tf.abs(target_Q - current_Q1)

    def _prioritized_experience_update(self, per, td_error, next_state, action, reward, replay_buffer):
        '''Updates the priorities in the PER buffer depending on the *per* int.
        :params per: 
        If 1, sample based on absolute TD-error.
        If 2, sample based on the distance between goal and state.
        If 3, sample proportional to the reward of a transition.
        If 4, sample transitions with a reward with a higher probability.
        N.B. Python doesn't have switch statements...'''
        if per: 
            if per == 1:
                error = td_error
            elif per == 2:
                error = 1 / (tf.norm(next_state[:,:action.shape[1]] - action, axis=1) + 0.00001)
            elif per == 3:
                error = reward + 1
            elif per == 4:
                # TODO replace -1 by c * -1 * meta_rew_scale
                error = np.where(reward == -1., 0, 1)
            replay_buffer.update_priorities(error)

    def _maybe_log_critic(self, gradients, norm, critic_loss, log):
        if log:
            self.cr_gr_norm.assign(norm)
            self.cr_gr_std.assign(tf.reduce_mean([tf.math.reduce_std(x) for x in gradients])) 
            self.critic_loss.assign(critic_loss)

    def _maybe_log_actor(self, gradients, norm, mean_actor_loss, log): 
        if log:
            self.ac_gr_norm.assign(norm)
            self.ac_gr_std.assign(tf.reduce_mean([tf.math.reduce_std(x) for x in gradients])) 
            self.actor_loss.assign(mean_actor_loss)

    def _create_persistent_tf_variables(self):
        # Create tf.Variables here. They persist the graph and can be used inside and outside as they hold their values.
        self.total_it = tf.Variable(0, dtype=tf.int32)         
        self.actor_loss = tf.Variable(0, dtype=tf.float32)
        self.critic_loss = tf.Variable(0, dtype=tf.float32)
        self.ac_gr_norm = tf.Variable(0, dtype=tf.float32)
        self.cr_gr_norm = tf.Variable(0, dtype=tf.float32)
        self.ac_gr_std = tf.Variable(0, dtype=tf.float32)
        self.cr_gr_std = tf.Variable(0, dtype=tf.float32)

    def _prepare_parameters(self, kwargs):
        kwargs = {f'_{key}': value for key, value in kwargs.items()}
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        self._subgoal_ranges = np.array(self._subgoal_ranges, dtype=np.float32)

    def _prepare_algo_objects(self):
        assert self._offpolicy != None
        assert self._name == 'meta' or self._name == 'sub'
        # Use these to change optimizer parameters mid-training
        self._ac_lr = tf.Variable(self._ac_lr)
        self._beta_1 = tf.Variable(0.9)
        self._beta_2 = tf.Variable(0.999)
        # Create networks 
        self._max_action  = tf.constant(self._max_action, dtype=tf.float32)
        self.actor = Actor(self._state_dim, self._action_dim, self._max_action, self._ac_hidden_layers, self._reg_coeff_ac)
        # Use reset nets to re-initialize networks at some point
        self.actor_reset_net = Actor(self._state_dim, self._action_dim, self._max_action, self._ac_hidden_layers,
                                     self._reg_coeff_ac)
        self.actor_target = Actor(self._state_dim, self._action_dim, self._max_action, self._ac_hidden_layers, self._reg_coeff_ac)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self._ac_lr, beta_1=self._beta_1, beta_2=self._beta_2)
        self.critic = Critic(self._state_dim, self._action_dim, self._cr_hidden_layers, self._reg_coeff_cr)
        self.critic_reset_net = Critic(self._state_dim, self._action_dim, self._cr_hidden_layers, self._reg_coeff_cr)
        self.critic_target = Critic(self._state_dim, self._action_dim, self._cr_hidden_layers, self._reg_coeff_cr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self._cr_lr, beta_1=self._beta_1, beta_2=self._beta_2)
        # Huber loss does not punish a noisy large gradient.
        self.critic_loss_fn = tf.keras.losses.Huber(delta=1.)  
        # Equal initial network and target network weights
        self.update_target_models_hard()  

if __name__ == '__main__':
 pass
