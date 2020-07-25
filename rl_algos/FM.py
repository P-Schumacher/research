import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from pudb import set_trace
from rl_algos.networks import ForwardModelNet
import wandb

class ForwardModel:
    def __init__(self, state_dim):
        self.net = ForwardModelNet(state_dim, [100], 0)
        self.opt = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.it = 0

    def get_reward(self, state, reshape=True):
        if reshape:
            state = tf.reshape(state, shape=[1, state.shape[0]])
        return self.net(state)

    def train(self, state, reward):
        state = tf.reshape(state, shape=[1, state.shape[0]])
        with tf.GradientTape() as tape:
            output = self.get_reward(state, reshape=False)
            loss = self.loss_fn(output, reward)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        if self.it < 100000000:
            self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
        self.log(np.abs(output - reward)/np.abs(reward), loss, output, reward)
        self.it += 1

    def log(self, pred_err, loss, y_pred, y_true):
        wandb.log({'FM/pred_error': pred_err, 'FM/loss': loss, 'FM/output': y_pred, 'FM/online_reward': y_true}, commit=False)

if __name__ == '__main__':
    pred_err = [] 
    model = ForwardModel(5)
    state = np.random.uniform(size=[1, 5])
    true_rew = 0.
    for i in range(1000):
        rew = model.get_reward(state)
        print(f'Prediction error:{np.abs(rew - true_rew)}')
        model.train(state, true_rew)
        pred_err.append(np.abs(rew - true_rew)[0])

    plt.plot(pred_err)
    plt.show()





