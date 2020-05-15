import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import wandb
from pudb import set_trace
from rl_algos.TD3_tf import TD3
from utils.math_fns import euclid, get_norm, clip_by_global_norm

class SplittedTD3(TD3):
    def train(self, replay_buffer, batch_size, t, log=False):
        state, action, reward, next_state, done, state_seq, action_seq = replay_buffer.sample(batch_size)
        td_error = self._train_step_critic(state, action, reward, next_state, done, log, replay_buffer.is_weight)
        self._prioritized_experience_update(self._per, td_error, next_state, action, reward,
                                     replay_buffer)

        wandb.log({'td_error_as_seen_by_critic': np.mean(td_error)}, commit=False)
        if self.total_it % self.policy_freq == 0:
            #state, action, reward, next_state, done, state_seq, action_seq = replay_buffer.sample_low(batch_size)
            td_error_as_seen_by_actor, error_before = self._train_step_actor(state, action, reward, next_state, done, log, replay_buffer.is_weight)
            #self._prioritized_experience_update(self._per, td_error_as_seen_by_actor, next_state, action, reward,
                                     #replay_buffer)
            wandb.log({'td_error_as_seen_by_actor': np.mean(error_before)}, commit=False)
        self.total_it.assign_add(1)
        if log:
            wandb.log({f'{self.name}/mean_weights_actor': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.actor.weights])}, commit=False)
            wandb.log({f'{self.name}/mean_weights_critic': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.critic.weights])}, commit=False)


        state, action, reward, next_state, done, state_seq, action_seq = replay_buffer.sample(batch_size)
        td_error = self._compute_td_errors(state, action, reward, next_state, done)
        self._prioritized_experience_update(self._per, td_error, next_state, action, reward,
                                     replay_buffer)
        return self.actor_loss.numpy(), self.critic_loss.numpy(), self.ac_gr_norm.numpy(), self.cr_gr_norm.numpy(), self.ac_gr_std.numpy(), self.cr_gr_std.numpy()

    def _compute_td_errors(self, state, action, reward, next_state, done):
        state_action = tf.concat([state, action], 1) # necessary because keras needs there to be 1 input arg to be able to build the model from shapes
        done = tf.reshape(done, [done.shape[0], 1])
        reward = tf.reshape(reward, [reward.shape[0], 1])
        noise = tf.random.normal(action.shape, stddev=self.policy_noise)
        # this clip keeps the noisy action close to the original action
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_state) + noise
        # this clip assures that we don't take impossible actions a > max_action
        next_action = tf.clip_by_value(next_action, -self._max_action, self._max_action)
        # Compute the target Q value
        next_state_next_action = tf.concat([next_state, next_action], 1)
        target_Q1, target_Q2 = self.critic_target(next_state_next_action)
        target_Q = tf.math.minimum(target_Q1, target_Q2)
        target_Q = reward + (1. - done) * self.discount * target_Q
        # Critic Update
        current_Q1, current_Q2 = self.critic(state_action)
        return tf.abs(target_Q - current_Q1)

    def _log_td_errors(self, replay_buffer, batch_size):
        errors = np.zeros([replay_buffer.size,])
        for i in range(replay_buffer.size):
            print(f'log {i} of {replay_buffer.size}')
            state = replay_buffer.state[i, :]
            state = np.reshape(state, [1, state.shape[0]])
            action = replay_buffer.action[i, :]
            action= np.reshape(action, [1, action.shape[0]])
            reward= replay_buffer.reward[i, :]
            reward= np.reshape(reward, [1, reward.shape[0]])
            next_state = replay_buffer.next_state[i, :]
            next_state = np.reshape(next_state, [1, next_state.shape[0]])
            done = replay_buffer.done[i, :]
            done= np.reshape(done, [1, done.shape[0]])
            error = self._compute_td_errors(state, action, reward, next_state, done)
            errors[i] = error
        wandb.log({'mean_td_error': np.mean(errors)},commit=False)

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
                error = tf.abs(td_error)
            replay_buffer.update_priorities(error)


    @tf.function
    def _train_step_critic(self, state, action, reward, next_state, done, log, is_weight):
        '''Training function. We assign actor and critic losses to state objects so that they can be easier recorded 
        without interfering with tf.function. I set Q terminal to 0 regardless if the episode ended because of a success cdt. or 
        a time limit. The norm and std of the updated gradients, as well as the losses are assigned to state objects of the class. 
        This is done as tf.function decorated functions are converted to static graphs and cannot handle variable return objects.
        :param : These should be explained by any Reinforcement Learning book.
        :return: None'''
        state_action = tf.concat([state, action], 1) # necessary because keras needs there to be 1 input arg to be able to build the model from shapes
        done = tf.reshape(done, [done.shape[0], 1])
        reward = tf.reshape(reward, [reward.shape[0], 1])
        noise = tf.random.normal(action.shape, stddev=self.policy_noise)
        # this clip keeps the noisy action close to the original action
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_state) + noise
        # this clip assures that we don't take impossible actions a > max_action
        next_action = tf.clip_by_value(next_action, -self._max_action, self._max_action)
        # Compute the target Q value
        next_state_next_action = tf.concat([next_state, next_action], 1)
        target_Q1, target_Q2 = self.critic_target(next_state_next_action)
        target_Q = tf.math.minimum(target_Q1, target_Q2)
        target_Q = reward + (1. - done) * self.discount * target_Q
        # Critic Update
        with tf.GradientTape() as tape:
            current_Q1, current_Q2 = self.critic(state_action)
            current_Q1 = current_Q1 * is_weight
            current_Q2 = current_Q2 * is_weight
            critic_loss = (self.critic_loss_fn(current_Q1, target_Q) 
                        + self.critic_loss_fn(current_Q2, target_Q))
            # 6 because Q losses + L2-regul losses
            assert len(self.critic.losses) == 6
            # critic.losses gives us the regularization losses from the layers
            critic_loss += sum(self.critic.losses)

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        gradients, norm = clip_by_global_norm(gradients, self.clip_cr)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        self._maybe_log_critic(gradients, norm, critic_loss, log)
        return target_Q - current_Q1

    @tf.function
    def _train_step_actor(self, state, action, reward, next_state, done, log, is_weight):
        # Can't use *if not* in tf.function graph
        # Actor update
        error_before = self._compute_td_errors(state, action, reward, next_state, done)
        with tf.GradientTape() as tape:
            action = self.actor(state)
            state_action = tf.concat([state, action], 1)
            actor_loss = self.critic.Q1(state_action)
            mean_actor_loss = -tf.math.reduce_mean(actor_loss)
        gradients = tape.gradient(mean_actor_loss, self.actor.trainable_variables)
        gradients, norm  = clip_by_global_norm(gradients, self.clip_ac)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
        self.transfer_weights(self.actor, self.actor_target, self.tau)
        self.transfer_weights(self.critic, self.critic_target, self.tau)
        self._maybe_log_actor(gradients, norm, mean_actor_loss, log) 
        error = self._compute_td_errors(state, action, reward, next_state, done)
        return error, error_before
