import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from pudb import set_trace
from rl_algos.networks import ForwardModelNet
#from networks import ForwardModelNet
import wandb

class ForwardModel:
    '''Model that learns the reward in tandem with the RL agent learning.'''
    def __init__(self, state_dim, logging, oracle=False):
        self.net = ForwardModelNet(2*state_dim, [100], 0)
        self.opt = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.logging = logging
        self.oracle = oracle
        self.reset(1000, 26)

    def reset(self, buffer_dim, state_dim):
        self.max_size = buffer_dim
        self.states = np.zeros([buffer_dim, 26], dtype=np.float32)
        self.next_states = np.zeros([buffer_dim, 26], dtype=np.float32)
        self.rewards = np.zeros([buffer_dim, 1], dtype=np.float32)
        self.dones= np.zeros([buffer_dim, 1], dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def sample(self, batch_size):
        batch_idxs = self._sample_idx(batch_size)
        return (
        tf.convert_to_tensor(self.states[batch_idxs]),
        tf.convert_to_tensor(self.next_states[batch_idxs]),
        tf.convert_to_tensor(self.rewards[batch_idxs]),
        tf.convert_to_tensor(self.dones[batch_idxs]))

    def add(self, state, next_state, reward, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _sample_idx(self, batch_size):
        return tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)

    def forward_pass(self, state, next_state, done, reshape=True, reversal=False):
        '''Computes estimated rewards and done in a forward pass.
        This information has then to be called from the class.'''
        if not self.oracle:
            reduced_state = tf.concat([state, next_state], axis=-1)
            out = self.net(reduced_state)
            return out[:,0, tf.newaxis], tf.cast(out[:,1, tf.newaxis] >= 20, dtype=tf.float32)
        else:
            reduced_state = tf.concat([state[:,-4, tf.newaxis], state[:, -1, tf.newaxis], next_state[:, -4, tf.newaxis],
                                       next_state[:, -1, tf.newaxis]], axis=-1)
            if reshape:
                reduced_state = tf.reshape(reduced_state, shape=[1, reduced_state.shape[-1]])
            return self.predict_oracle(reduced_state, done, reversal)

    def get_reward(self):
        return self.reward

    def get_done(self):
        done = self.done >= 20
        return done

    def predict_oracle(self, state, done, reversal):
        if not reversal:
            ret = tf.constant([49. if np.all(x == [1.,-1,1,1]) else -1. for x in state])[:, tf.newaxis]
        else:
            ret = tf.constant([49. if np.all(x == [-1.,1,1,1]) else -1. for x in state])[:, tf.newaxis]
        if reversal:
            return ret, done * -1. + 1. 
        else:
            return ret, done 

    def train(self, state, next_state, reward, done, reversal=False):
        self.add(state, next_state, reward, done)
        states, next_states, rewards, dones  = self.sample(64)
        pred_err, loss, y_pred, y_true = self._train(states, next_states, rewards, dones)
        if self.logging:
            self.log(tf.reduce_mean(pred_err), tf.reduce_mean(loss))

    def log(self, pred_err, loss):
        wandb.log({'FM/pred_error': pred_err, 'FM/loss': loss}, commit=False)
        wandb.log({f'FM/mean_weights': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.net.weights])}, commit=False)

    def _train(self, state, next_state, reward, done, reversal=False):
        with tf.GradientTape() as tape:
            ret_pred, done_pred = self.forward_pass(state, next_state, done, reshape=False, reversal=reversal)
            loss = self.loss_fn(ret_pred, reward) + self.loss_fn(done_pred, 50*done) 
        gradients = tape.gradient(loss, self.net.trainable_variables)
        if not self.oracle:
            self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
        return tf.abs(ret_pred-reward)/tf.abs(reward), loss, ret_pred, reward

if __name__ == '__main__':
    pred_err = [] 
    model = ForwardModel(2, logging=False)
    states = np.array([[0.,0,0,0], [1,0,1,0],[1,0,1,1],[0,1,0,1],[0,1,1,1], [0,0,1,0],[0,0,0,1]], dtype=np.float32)
    true_rews = np.array([[-1.],[-1],[50], [-1],[-1], [-1], [-1]], dtype=np.float32)
    for i in range(100):
        for concatstate, reward in zip(states, true_rews):
            choice = np.random.randint(0, 7)
            print(choice)
            concatstate = states[choice, :]
            reward = true_rews[choice, :]
            state = np.zeros([1, 26])
            next_state = np.zeros([1, 26])
            state[:, -4] = concatstate[0]
            state[:, -1] = concatstate[1]
            next_state[:, -4] = concatstate[2]
            next_state[:, -1] = concatstate[3]
            rew_pred = model.get_reward(state, next_state, reshape=False)
            print(f'Prediction error:{np.abs(rew_pred - reward)}')
            model.train(state, next_state, reward)
            pred_err.append(np.abs(rew_pred - reward)[0])
    plt.plot(pred_err)
    preds = []
    for concatstate, reward in zip(states, true_rews):
        state = np.zeros([1, 26])
        next_state = np.zeros([1, 26])
        state[:, -4] = concatstate[0]
        state[:, -1] = concatstate[1]
        next_state[:, -4] = concatstate[2]
        next_state[:, -1] = concatstate[3]
        rew_pred = model.get_reward(state, next_state, reshape=False)
        preds.append(rew_pred[0])
    print([x.numpy() for x in preds])
    plt.show()





