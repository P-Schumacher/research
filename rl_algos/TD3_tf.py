import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as inits
import numpy as np
from tensorflow.keras.regularizers import l2
import wandb
from pudb import set_trace
from utils.math_fns import euclid, get_norm, clip_by_global_norm
from rl_algos.networks import Actor, Critic
from rl_algos.offpol_correction import off_policy_correction


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        subgoal_ranges,
        target_dim,
        ac_hidden_layers,
        cr_hidden_layers,
        clip_cr,
        clip_ac,
        reg_coeff_ac,
        reg_coeff_cr,
        zero_obs,
        per,
        goal_regul,
        name="default",
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        ac_lr = 0.0001,
        cr_lr = 0.01,
        offpolicy = None,
        c_step = 10,
        no_candidates = 10
    ):
        assert offpolicy != None
        assert name == 'meta' or name == 'sub'
        # Create networks 
        max_action  = tf.constant(max_action, dtype=tf.float32)
        self.actor = Actor(state_dim, action_dim, max_action, ac_hidden_layers, reg_coeff_ac)
        self.actor_target = Actor(state_dim, action_dim, max_action, ac_hidden_layers, reg_coeff_ac)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ac_lr)
        self.critic = Critic(state_dim, action_dim, cr_hidden_layers, reg_coeff_cr)
        self.critic_target = Critic(state_dim, action_dim, cr_hidden_layers, reg_coeff_cr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=cr_lr)
        # Huber loss does not punish a noisy large gradient.
        self.critic_loss_fn = tf.keras.losses.Huber(delta=1.)  
        # Equal initial network and target network weights
        self.update_target_models_hard()  

        self._prepare_parameters(name, offpolicy, max_action, discount, tau, policy_noise, noise_clip, policy_freq,
                                 c_step, no_candidates, subgoal_ranges, target_dim, clip_cr, clip_ac, zero_obs, per,
                                 goal_regul)

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
     
    def train(self, replay_buffer, batch_size, t, log=False, sub_actor=None):
        state, action, reward, next_state, done, state_seq, action_seq = replay_buffer.sample(batch_size)
        if self.offpolicy and self.name == 'meta': 
            action = off_policy_correction(self.subgoal_ranges, self.target_dim, sub_actor, action, state, next_state, self.no_candidates,
                                          self.c_step, state_seq, action_seq, self._zero_obs)
        if self.name == 'meta' and self._goal_regul:
            reward_new = self._goal_regularization(action, reward, next_state)
        else:
            reward_new = reward
        td_error = self._train_critic(state, action, reward_new, next_state, done, log, replay_buffer.is_weight)
        self._prioritized_experience_update(self._per, td_error, next_state, action, reward, replay_buffer)
        #state, action, reward, next_state, done, state_seq, action_seq = replay_buffer.sample_low(batch_size)
        self._train_actor(state, action, reward_new, next_state, done, log, replay_buffer.is_weight)
        #td_error = self._compute_td_error(state, action, reward, next_state, done)
        #self._prioritized_experience_update(self._per, td_error, next_state, action, reward, replay_buffer)
        self.total_it.assign_add(1)


        if log:
            wandb.log({f'{self.name}/mean_weights_actor': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.actor.weights])}, commit=False)
            wandb.log({f'{self.name}/mean_weights_critic': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.critic.weights])}, commit=False)

        return self.actor_loss.numpy(), self.critic_loss.numpy(), self.ac_gr_norm.numpy(), self.cr_gr_norm.numpy(), self.ac_gr_std.numpy(), self.cr_gr_std.numpy()

    @tf.function
    def _compute_td_error(self, state, action, reward, next_state, done):
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
    def _train_actor(self, state, action, reward_new, next_state, done, log, is_weight):
        # Can't use *if not* in tf.function graph
        if self.total_it % self.policy_freq == 0:
            # Actor update
            with tf.GradientTape(persistent=True) as tape:
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

    def _goal_regularization(self, action, reward, next_state):
        ans = reward - self._goal_regul * euclid(next_state[:, :action.shape[1]] - action)
        ans = reward - tf.reshape(self._goal_regul * euclid(next_state[:, :action.shape[1]] - action, axis=1), [128,1])
        return ans        

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
                error = 1/(tf.abs(td_error)+0.000001)
                #error = tf.abs(td_error)
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

    def _prepare_parameters(self, name, offpolicy, max_action, discount, tau, policy_noise, noise_clip, policy_freq,
                            c_step, no_candidates, subgoal_ranges, target_dim, clip_cr, clip_ac, zero_obs, per,
                            goal_regul):
        # Save parameters
        self.name = name
        self.offpolicy = offpolicy
        self._max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.c_step = c_step
        self.no_candidates = no_candidates 
        self.subgoal_ranges = np.array(subgoal_ranges, dtype=np.float32)
        self.target_dim = target_dim
        self.clip_cr = clip_cr
        self.clip_ac = clip_ac
        self._zero_obs = zero_obs
        self._per = per
        self._goal_regul = goal_regul


if __name__ == '__main__':
 pass
