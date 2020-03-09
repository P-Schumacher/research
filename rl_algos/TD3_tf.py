import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as inits
import numpy as np
import copy
from matplotlib import pyplot as plt
from pudb import set_trace
import datetime
from tensorflow.keras.regularizers import l2
import wandb

initialize_relu = inits.VarianceScaling(scale=1./3., mode="fan_in", distribution="uniform")  # this conserves std for layers with relu activation 
initialize_tanh = inits.GlorotUniform()  # This is the standard tf.keras.layers.Dense initializer, it conserves std for layers with tanh activation

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action, ac_layers, reg_coeff):
        super(Actor, self).__init__()

        self.l1 = kl.Dense(ac_layers[0], activation='relu', kernel_initializer=initialize_relu)
        self.l2 = kl.Dense(ac_layers[1], activation='relu', kernel_initializer=initialize_relu)
        self.l3 = kl.Dense(action_dim, activation='tanh', kernel_initializer=initialize_tanh)
        
        self.max_action = max_action
        # Remember building your model before you can copy it
        # else the weights wont be there. Could also call the model once in the beginning to build it implicitly 
        self.build(input_shape=(None, state_dim))

    @tf.function 
    def call(self, state):
        assert state.dtype == tf.float32
        x = self.l1(state)
        x = self.l2(x)
        return self.max_action * self.l3(x)


class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, cr_layers, reg_coeff):
        super(Critic, self).__init__()
        # Q1 architecture 
        self.l1 = kl.Dense(cr_layers[0], activation='relu', kernel_initializer=initialize_relu,
                           kernel_regularizer=l2(reg_coeff))
        self.l2 = kl.Dense(cr_layers[1], activation='relu', kernel_initializer=initialize_relu,
                           kernel_regularizer=l2(reg_coeff))
        self.l3 = kl.Dense(1, 
                           kernel_regularizer=l2(reg_coeff))

        # Q2 architecture
        self.l4 = kl.Dense(300, activation='relu', kernel_initializer=initialize_relu,
                           kernel_regularizer=l2(reg_coeff))
        self.l5 = kl.Dense(300, activation='relu', kernel_initializer=initialize_relu,
                           kernel_regularizer=l2(reg_coeff))
        self.l6 = kl.Dense(1, kernel_regularizer=l2(reg_coeff))
        self.build(input_shape=(None, state_dim+action_dim))

    @tf.function
    def call(self, state_action):
        assert state_action.dtype == tf.float32
        q1 = self.l1(state_action) # activation fcts are build-in the layer constructor
        q1 = self.l2(q1) 
        q1 = self.l3(q1)
        
        q2 = self.l4(state_action)
        q2 = self.l5(q2)
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state_action):
        q1 = self.l1(state_action)
        q1 = self.l2(q1)
        q1 = self.l3(q1)
        return q1


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
        self.critic_loss_fn = tf.keras.losses.Huber(delta=1.)  # Huber loss does not punish a noisy large gradient.
        
        self.update_target_models_hard()  # Equal network and target network weights
        # Save parameters
        self.name = name
        self.offpolicy = offpolicy
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.c_step = c_step
        self.no_candidates = no_candidates 
        self.subgoal_ranges = subgoal_ranges
        self.target_dim = target_dim
        self.clip_cr = clip_cr
        self.clip_ac = clip_ac
        # Create tf.Variables here. They persist the graph and can be used inside and outside as they hold their values.
        self.total_it = tf.Variable(0, dtype=tf.int64)         
        self.actor_loss = tf.Variable(0, dtype=tf.float32)
        self.critic_loss = tf.Variable(0, dtype=tf.float32)
        self.ac_gr_mean = tf.Variable(0, dtype=tf.float32)
        self.cr_gr_mean = tf.Variable(0, dtype=tf.float32)
        self.ac_gr_std = tf.Variable(0, dtype=tf.float32)
        self.cr_gr_std = tf.Variable(0, dtype=tf.float32)

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
     
    def train(self, replay_buffer, batch_size, t, sub_actor=None):
        state, action, next_state, reward, done, state_seq, action_seq = replay_buffer.sample(batch_size)
        
        if self.offpolicy and self.name == 'meta': 
            action = off_policy_correction(self.subgoal_ranges, self.target_dim, sub_actor, action, state, next_state, self.no_candidates,
                                          self.c_step, state_seq, action_seq)
        self.train_step(state, action, next_state, reward, done)
        self.total_it.assign_add(1)
        return self.actor_loss.numpy(), self.critic_loss.numpy(), self.ac_gr_mean.numpy(), self.cr_gr_mean.numpy(), self.ac_gr_std.numpy(), self.cr_gr_std.numpy()
   
    @tf.function
    def train_step(self, state, action, next_state, reward, done):
        '''Training function. We assign actor and critic losses to sate objects so that they can be easier plotted
        without interfering with tf.function'''
        state_action = tf.concat([state, action], 1) # necessary because keras needs there to be 1 input arg to be able to build the model from shapes
        noise = tf.random.normal(action.shape, stddev=self.policy_noise)
        # this clip keeps the noisy action close to the original action
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_state) + noise
        # this clip assures that we don't take impossible actions a > max_action
        next_action = tf.clip_by_value(next_action, -self.max_action, self.max_action)
        # Compute the target Q value
        next_state_next_action = tf.concat([next_state, next_action], 1)
        target_Q1, target_Q2 = self.critic_target(next_state_next_action)
        target_Q = tf.math.minimum(target_Q1, target_Q2)
        target_Q = reward + (1. - done) * self.discount * target_Q
        with tf.GradientTape() as tape:
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state_action)
            # Sum of losses allows us to calculate them independantly at the same time,
            # as gradient is a linear operation
            critic_loss = (self.critic_loss_fn(current_Q1, target_Q) 
                        + self.critic_loss_fn(current_Q2, target_Q))
            assert len(self.critic.losses) == 6
            critic_loss += sum(self.critic.losses)

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        #gradients = [tf.clip_by_norm(grad, self.clip_cr) for grad in gradients]
        gradients = tf.clip_by_global_norm([grad for grad in gradients], self.clip_cr)
        self.cr_gr_mean.assign(tf.reduce_mean([tf.reduce_mean(x) for x in gradients]))
        self.cr_gr_std.assign(tf.reduce_mean([tf.math.reduce_std(x) for x in gradients])) 
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        self.critic_loss.assign(critic_loss)
        if self.total_it % self.policy_freq == 0:
            with tf.GradientTape() as tape:
            # The gradient of Q_theta_1 w.r.t. phi (the actor weights)
            # is equal to the product of the gradient of Q_theta_1 w.r.t. the actions and 
            # the gradient of the actor w.r.t. phi 
            # Look at TD3 paper for clarification
                action = self.actor(state)
                state_action = tf.concat([state, action], 1)
                actor_loss = self.critic.Q1(state_action)
                mean_actor_loss = -tf.math.reduce_mean(actor_loss)

            gradients = tape.gradient(mean_actor_loss, self.actor.trainable_variables)
            #gradients = [tf.clip_by_norm(grad, self.clip_ac) for grad in gradients]
            gradients = tf.clip_by_global_norm([grad for grad in gradients], self.clip_ac)
            self.ac_gr_mean.assign(tf.reduce_mean([tf.reduce_mean(x) for x in gradients]))
            self.ac_gr_std.assign(tf.reduce_mean([tf.math.reduce_std(x) for x in gradients])) 
            self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
            self.transfer_weights(self.actor, self.actor_target, self.tau)
            self.transfer_weights(self.critic, self.critic_target, self.tau)
            self.actor_loss.assign(mean_actor_loss)

@tf.function
def off_policy_correction(subgoal_ranges, target_dim, pi, goal_b, state_b, next_state_b, no_candidates, c_step, state_seq,
                          action_seq):
    # TODO Update docstring to real dimensions
    '''Computes the off-policy correction for the meta-agent.
    c = candidate_nbr; t = time; i = vector coordinate (e.g. action 0 of 8 dims) 
    Dim(candidates) = [c i]
    Dim(state_b)     = [t i]
    Dim(action_seq)    = [t i]
    Dim(prop_goals) = [c_step, b_size, g_dim, no_candidates]'''
    
    b_size = state_b.shape[0] # Batch Size
    g_dim = goal_b[0].shape[0] # Subgoal dimension
    action_dim = action_seq.shape[-1] # action dimension
    # States contains state+target. Need to take out only the state.
    state_seq = state_seq[:,:, :-target_dim]
    state_b = state_b[:, :-target_dim]
    next_state_b = next_state_b[:, :-target_dim]
    # Get the sampled candidates
    candidates =  get_goal_candidates(b_size, g_dim, no_candidates, subgoal_ranges, state_b, next_state_b, goal_b)
    # Take all the possible candidates and propagate them through time using h = st + gt - st+1 cf. HIRO Paper
    prop_goals = multi_goal_transition(state_seq, candidates, c_step)
    # Zero out xy for sub agent, AFTER goals have been calculated from it.
    set_trace()
    state_seq = tf.reshape(state_seq, [b_size * c_step, state_seq.shape[-1]])
    state_seq *= tf.concat([tf.zeros([state_seq.shape[0], 2]), tf.ones([state_seq.shape[0], state_seq.shape[1] - 2])], axis=1)
    best_c = get_best_c(b_size, c_step, action_dim, g_dim, no_candidates, action_seq, state_seq, prop_goals, pi) 
    return get_corrected_goal(b_size, candidates, best_c) 

def multi_goal_transition(state_seq, candidates, c_step):
    # Realize that the multi timestep goal transition can be vectorized to a single calculation.
    b_size = candidates.shape[0]
    g_dim = candidates.shape[1]
    c_step = state_seq.shape[1] 
    no_candidates = candidates.shape[2]
    # In state_seq, equal batches are separated by c timesteps if we query a subogoal every c_steps 
    prop_goals = tf.broadcast_to(tf.expand_dims(candidates, axis=1), [b_size, c_step, g_dim, no_candidates])
    tmp = tf.broadcast_to(tf.expand_dims(state_seq[:,0,:g_dim], axis=1), [b_size, c_step, g_dim]) - state_seq[:, :, :g_dim] 
    prop_goals += tf.broadcast_to(tf.expand_dims(tmp, axis=3), [b_size, c_step, g_dim, no_candidates]) 
    return prop_goals 

def get_goal_candidates(b_size, g_dim, no_candidates, subgoal_ranges, state_b, next_state_b, goal_b):
    # Original Goal
    orig_goal = tf.expand_dims(goal_b[:, :g_dim], axis=2)
    # Goal between the states s_t+1 - s_t
    diff_goal = tf.expand_dims(next_state_b[:, :g_dim] - state_b[:, :g_dim], axis=2)
    goal_mean = tf.broadcast_to(diff_goal, [b_size, g_dim, no_candidates])
    # Broadcast the subgoal_ranges to [b_size, g_dim, no_candidates] for use as clipping and as std
    clip_tensor = tf.expand_dims(tf.broadcast_to(subgoal_ranges, [b_size, subgoal_ranges.shape[0]]), axis=2)
    clip_tensor = tf.broadcast_to(clip_tensor, [b_size, subgoal_ranges.shape[0], no_candidates+2])
    goal_std = 0.25 * clip_tensor # cf. HIRO Paper
    candidates = tf.random.normal([b_size, g_dim, no_candidates], goal_mean, goal_std[:, :, :no_candidates])
    candidates = tf.concat([orig_goal, diff_goal, candidates], axis=2)
    candidates = tf.clip_by_value(candidates, -clip_tensor, clip_tensor)
    return candidates

def get_corrected_goal(b_size, candidates, best_c): 
    corrected_goals = tf.TensorArray(dtype=tf.float32, size=b_size)
    # TODO Does this need a loop?
    for b in tf.range(b_size):
        corrected_goals = corrected_goals.write(b, candidates[b, :, best_c[b]])
    return corrected_goals.stack() 

def get_best_c(b_size, c_step, action_dim, g_dim, no_candidates, action_seq, state_seq, prop_goals, pi):
    # Compute the logpropabilities for different subgoal candidates
    max_logprob = tf.constant(-1e9, shape=[b_size,])
    best_c = tf.zeros([b_size,], dtype=tf.int32) # In Graph mode, elements need to be defined BEFORE the loop
    action_seq = tf.reshape(action_seq, [b_size * c_step, action_dim])
    for c in tf.range(no_candidates):
        goals = prop_goals[:, :, :, c]  # Pick one candidate 
        # Change dimension of goals so that it can be computed in one batch in the network [b_size * c_step, g_dim]
        # TODO reshape prop_goals before the loop
        goals = tf.reshape(goals, [b_size * c_step, g_dim])
        state_action = tf.concat([state_seq, goals], axis=1)
        # TODO can we gather scatter this such that we dont compute infinities?
        pred_action =  pi(state_action)
        diff = action_seq - pred_action
        # Have padded the sequences where len(seq)<c_step to the max(c_step) with np.inf
        # This results in nan in the computed action differences. These are now set to 0
        # as 0 does not influence the sum over all actions that happens later.
        diff = tf.where(tf.math.is_nan(diff), 0., diff)
        # This reshape only works when rehaping consecutive dimension (e.g. [2560,8] to [256,10,8] or vice versa. 
        # Else elements will be thrown around.
        diffn = tf.reshape(diff, [b_size, c_step, action_dim])
        # cf. HIRO Paper
        logprob = - 0.5 * tf.reduce_sum(tf.square(tf.norm(diffn, axis=2)), axis=1)
        logprob = tf.ones([b_size,])
        best_c = tf.where(logprob > max_logprob, c, best_c)
        max_logprob = tf.where(logprob > max_logprob, logprob, max_logprob)
    return best_c

