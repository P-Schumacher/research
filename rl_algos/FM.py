import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from pudb import set_trace
from rl_algos.networks import ForwardModelNet
from collections import deque
#from networks import ForwardModelNet
import wandb

class ForwardModel:
    '''Model that learns the reward in tandem with the RL agent learning.'''
    def __init__(self, state_dim, logging, oracle=False, nstep=10):
        self.net = ForwardModelNet(2*state_dim, [100], 0.)
        self.opt = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.logging = logging
        self.oracle = oracle
        self.reset(1000, 26)
        self.nstep = nstep
        self.n_step_buffer = deque(maxlen=self.nstep)

    def reset(self, buffer_dim, state_dim):
        self.max_size = buffer_dim
        self.states = np.zeros([buffer_dim, 26], dtype=np.float32)
        self.next_states = np.zeros([buffer_dim, 26], dtype=np.float32)
        self.rewards = np.zeros([buffer_dim, 1], dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.t = 0

    def sample(self, batch_size):
        batch_idxs = self._sample_idx(batch_size)
        return (
        tf.convert_to_tensor(self.states[batch_idxs]),
        tf.convert_to_tensor(self.next_states[batch_idxs]),
        tf.convert_to_tensor(self.rewards[batch_idxs]))

    def add(self, state, next_state, reward, done, reset):
        '''Constructs multi-step transitions from one-step transitions
        to train more efficiently for the slow timescale policy
        in an HRL agent.
        Pay attention that we need to reset the internel multi-step
        buffer when the environment ends an episode, because
        we don't want multi-step transitions over an episode
        restart. This is different from the done parameter
        which indicates if the MDP ended.'''
        self.n_step_buffer.append((state, next_state, reward))
        if len(self.n_step_buffer) == self.nstep:
            state, next_state, reward,  = self._calc_multistep_transitions()
            self._add(state, next_state, reward)
            if done:
                self._add_non_strict_terminal_transitions()
            if reset:
                self.n_step_buffer.clear()

    def _add_non_strict_terminal_transitions(self):
        '''Instead of just adding {s_t, s_t+n} when the episode ends,
        we also add {s_t+1, s_t+n} ... {s_t+n-1, s_t+n} to obtain more
        information out of terminal transitions. 
        cf. Maitre Wilmot'''
        rew = 0
        for k in range(1, self.nstep):
            state = self.n_step_buffer[-k][0]
            next_state = self.n_step_buffer[-1][1]
            rew += self.n_step_buffer[-k][2]
            self._add(state, next_state, rew)

    def _add(self, state, next_state, reward):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _sample_idx(self, batch_size):
        return tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)

    def forward_pass(self, state, next_state, reshape=True, reversal=False):
        '''Computes estimated rewards and done in a forward pass.
        This information has then to be called from the class.'''
        if not self.oracle:
            reduced_state = tf.concat([state, next_state], axis=-1)
            return self.net(reduced_state)
        else:
            reduced_state = tf.concat([state[:,-4, tf.newaxis], state[:, -1, tf.newaxis], next_state[:, -4, tf.newaxis],
                                       next_state[:, -1, tf.newaxis]], axis=-1)
            if reshape:
                reduced_state = tf.reshape(reduced_state, shape=[1, reduced_state.shape[-1]])
            return self.predict_oracle(reduced_state, reversal)

    def predict_oracle(self, state, reversal):
        if not reversal:
            ret = tf.constant([49. if np.all(x == [1.,-1,1,1]) else -1. for x in state])[:, tf.newaxis]
        else:
            ret = tf.constant([49. if np.all(x == [-1.,1,1,1]) else -1. for x in state])[:, tf.newaxis]
        if reversal:
            return ret
        else:
            return ret

    def train(self, state, next_state, reward, done, reset, reversal=False):
        self.add(state, next_state, reward, done, reset)
        if self.size >= 500:
            states, next_states, rewards  = self.sample(128)
            high_prederr, low_prederr, loss, y_pred, y_true = self._train(states, next_states, rewards)
            if self.logging:
                self.log(high_prederr, low_prederr, loss)

    def log(self, high_prederr, low_prederr, loss):
        wandb.log({'FM/high_prederror': high_prederr, 'FM/low_prederror': low_prederr, 'FM/loss': loss}, commit=False)
        wandb.log({f'FM/mean_weights': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.net.weights])}, commit=False)

    #@tf.function
    def _train(self, state, next_state, reward, reversal=False):
        with tf.GradientTape() as tape:
            ret_pred = self.forward_pass(state, next_state, reshape=False, reversal=reversal)
            loss = self.loss_fn(ret_pred, reward) + sum(self.net.losses)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        if not self.oracle:
            self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
        high_rews = tf.where(reward != -10.)[:,0]
        low_rews = tf.where(reward == -10.)[:,0]
        high_preds = tf.gather(ret_pred, high_rews)
        low_preds = tf.gather(ret_pred, low_rews)
        high_rews = tf.gather(reward, high_rews)
        low_rews = tf.gather(reward, low_rews)
        return tf.reduce_sum(tf.abs(high_preds - high_rews)), tf.reduce_sum(tf.abs(low_preds - low_rews)), tf.reduce_sum(loss), ret_pred, reward

    def _calc_multistep_transitions(self):
        ret = 0
        for idx in range(self.nstep):
            ret +=  self.n_step_buffer[idx][2]
        return self.n_step_buffer[0][0], self.n_step_buffer[-1][1], ret

if __name__ == '__main__':
    pred_err = [] 
    model = ForwardModel(26, logging=False, nstep=1)
    states = np.array([[0.,0,0,0], [1,0,1,0],[1,0,1,1],[0,1,0,1],[0,1,1,1], [0,0,1,0],[0,0,0,1]], dtype=np.float32)
    true_rews = np.array([[-1.],[-1],[50], [-1],[-1], [-1], [-1]], dtype=np.float32)
    for i in range(1000):
        for concatstate, reward in zip(states, true_rews):
            choice = np.random.randint(0, 7)
            print(choice)
            #concatstate = states[choice, :]
            #reward = true_rews[choice, :]
            state = np.zeros([1, 26])
            next_state = np.zeros([1, 26])
            state[:, -4] = concatstate[0]
            state[:, -1] = concatstate[1]
            next_state[:, -4] = concatstate[2]
            next_state[:, -1] = concatstate[3]
            rew_pred = model.forward_pass(state, next_state, reshape=False)
            print(f'Prediction error:{np.abs(rew_pred - reward)}')
            model.train(state, next_state, reward)
            pred_err.append(np.abs(rew_pred - reward)[0])
    plt.subplot(121)
    plt.plot(pred_err)
    preds = []
    for concatstate, reward in zip(states, true_rews):
        state = np.zeros([1, 26])
        next_state = np.zeros([1, 26])
        state[:, -4] = concatstate[0]
        state[:, -1] = concatstate[1]
        next_state[:, -4] = concatstate[2]
        next_state[:, -1] = concatstate[3]
        rew_pred = model.forward_pass(state, next_state, reshape=False)
        preds.append(rew_pred[0])
    x = [x.numpy() for x in preds]
    print(x)
    plt.subplot(122)
    plt.plot(x)
    plt.show()





