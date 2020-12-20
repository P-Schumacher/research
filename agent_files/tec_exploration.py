import tensorflow as tf
from pudb import set_trace
from tensorflow.keras import layers, losses
from agent_files import HIRO
import hydra

class AutoEncoder(tf.keras.Model):

    def __init__(self, latent, state_dim, **cnf):
        super(AutoEncoder, self).__init__()
        for key in cnf.keys():
            setattr(self, key, cnf[key])
        self.encoder = layers.Dense(latent, activation=self.encod_activ)
        self.dropout_layer = layers.Dropout(self.dropout) #0.15
        self.decoder = layers.Dense(self.packet*state_dim, activation=self.decod_activ)
        self.reshape = layers.Reshape((self.packet, state_dim))
        self.flatten = layers.Flatten()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.lam = self.contractive #0.0005
        self.contractive = bool(self.contractive == 0.)
        self.train = tf.Variable(False, trainable=False, dtype=tf.bool)

    @tf.function
    def call(self, x):
        encoded = self.flatten(x)
        encoded = self.encoder(encoded)
        encoded = self.dropout_layer(encoded, training=self.train)
        decoded = self.decoder(encoded)
        decoded = self.reshape(decoded)
        return decoded, encoded

    def train_epoch(self, dataset, N):
        self.train.assign(True)
        for _ in range(N):
            for x, y in dataset:
                self.train_step(x, y)
        self.train.assign(False)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            re_x, hidden = self.__call__(x)
            batch_loss = self.loss(re_x, y, hidden)
        grad = tape.gradient(batch_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def loss(self, y_pred, y_true, hidden=None):
        if self.lam:
            return self.mse_loss(y_pred, y_true)
        else:
            return self.contractive_loss(y_pred, y_true, hidden)

    def contractive_loss(self, y_pred, y_true, hidden):
        W = tf.transpose(self.encoder.weights[0])
        mse = self.mse_loss(y_pred, y_true)
        dh = hidden * (1 - hidden)
        contract = tf.reduce_sum(tf.square(W), axis=1)
        contract = self.lam * tf.reduce_sum(tf.square(dh) * contract, axis=1)
        return mse + contract 


class TecAgent(HIRO.HierarchicalAgent):
    def __init__(self, agent_cnf, buffer_cnf, main_cnf, env_spec, model_cls, subgoal_dim, tec_cnf):
        super(TecAgent, self).__init__(agent_cnf, buffer_cnf, main_cnf, env_spec, model_cls, subgoal_dim)
        self.lossy_ae = AutoEncoder(40, env_spec['state_dim'], **tec_cnf)
        print(self.lossy_ae)

         

