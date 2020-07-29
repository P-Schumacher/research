import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from pudb import set_trace
from rl_algos.networks import ForwardModelNet
#from networks import ForwardModelNet
import wandb

class ForwardModel:
    def __init__(self, state_dim, logging):
        self.net = ForwardModelNet(2*state_dim, [100], 0)
        self.opt = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.logging = logging

    def get_reward(self, state, next_state, reshape=True):
        reduced_state = tf.concat([state[:,-4, tf.newaxis], state[:, -1, tf.newaxis], next_state[:, -4, tf.newaxis],
                                   next_state[:, -1, tf.newaxis]], axis=-1)
        if reshape:
            reduced_state = tf.reshape(reduced_state, shape=[1, reduced_state.shape[-1]])
        return self.net(reduced_state)

    def train(self, state, next_state, reward):
        pred_err, loss, y_pred, y_true = self._train(state, next_state, reward)
        if self.logging:
            self.log(pred_err, loss, y_pred, y_true)

    def log(self, pred_err, loss, y_pred, y_true):
        wandb.log({'FM/pred_error': pred_err, 'FM/loss': loss, 'FM/output': y_pred, 'FM/online_reward': y_true}, commit=False)

    @tf.function
    def _train(self, state, next_state, reward):
        state = tf.reshape(state, shape=[1, state.shape[0]])
        next_state = tf.reshape(next_state, shape=[1, next_state.shape[0]])
        with tf.GradientTape() as tape:
            output = self.get_reward(state, next_state, reshape=True)
            loss = self.loss_fn(output, reward)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
        return tf.abs(output-reward)/tf.abs(reward), loss, output, reward

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





