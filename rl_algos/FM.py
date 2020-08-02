import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from pudb import set_trace
from rl_algos.networks import ForwardModelNet
#from networks import ForwardModelNet
import wandb

class ForwardModel:
    def __init__(self, state_dim, logging):
        self.net = ForwardModelNet(2*state_dim, [2], 0)
        self.opt = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.logging = logging
        self.reset(1000, 26)
        #self.add_manual([-1,-1,-1,-1,-1])
        #self.add_manual([-1,-1,1,-1,-1])
        #self.add_manual([-1,-1,-1,1,-1])
        #self.add_manual([1,-1,1,-1,-1])
        #self.add_manual([1,-1,1,1, 1])
        #self.add_manual([-1,1,-1,1,-1])
        #self.add_manual([-1,1,1,1,-1])

    def add_manual(self, vector):
            assert len(vector) == 5
            state = np.zeros(shape=[1,26])
            next_state = np.zeros(shape=[1,26])
            state[0,-4] = vector[0]
            state[0, -1] = vector[1]
            next_state[0, -4] = vector[2]
            next_state[0, -1] = vector[3]
            reward = vector[4]
            self.add(state, next_state, reward)

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

    def add(self, state, next_state, reward):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _sample_idx(self, batch_size):
        return tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)

    def get_reward(self, state, next_state, reshape=True):
        reduced_state = tf.concat([state[:,-4, tf.newaxis], state[:, -1, tf.newaxis], next_state[:, -4, tf.newaxis],
                                   next_state[:, -1, tf.newaxis]], axis=-1)
        if reshape:
            reduced_state = tf.reshape(reduced_state, shape=[1, reduced_state.shape[-1]])
        return self.net(reduced_state)

    def train(self, state, next_state, reward):
        self.add(state, next_state, reward)
        states, next_states, rewards = self.sample(64)
        pred_err, loss, y_pred, y_true = self._train(states, next_states, rewards)
        if self.logging:
            self.log(tf.reduce_mean(pred_err), tf.reduce_mean(loss))

    def log(self, pred_err, loss):
        wandb.log({'FM/pred_error': pred_err, 'FM/loss': loss}, commit=False)
        wandb.log({f'FM/mean_weights': wandb.Histogram([tf.reduce_mean(x).numpy() for x in self.net.weights])}, commit=False)

    @tf.function
    def _train(self, state, next_state, reward):
        with tf.GradientTape() as tape:
            output = self.get_reward(state, next_state, reshape=False)
            loss = self.loss_fn(output, reward)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
        return tf.abs(output-reward)/tf.abs(reward), loss, output, reward

if __name__ == '__main__':
    pred_err = [] 
    model = ForwardModel(2, logging=False)
    states = np.array([[0.,0,0,0], [1,0,1,0],[1,0,1,1],[0,1,0,1],[0,1,1,1], [0,0,1,0],[0,0,0,1]], dtype=np.float32)
    true_rews = np.array([[-1.],[-1],[1], [-1],[-1], [-1], [-1]], dtype=np.float32)
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





