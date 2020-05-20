from tensorflow import tf
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as inits
from tensorflow.keras.regularizers import l2

# this conserves std for layers with relu activation 
initialize_relu = inits.VarianceScaling(scale=1./3., mode="fan_in", distribution="uniform")  
# This is the standard tf.keras.layers.Dense initializer, it conserves std for layers with tanh activation
initialize_tanh = inits.GlorotUniform()  

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action, ac_layers, reg_coeff):
        '''Generic actor class for an actor-critic algorithm. 
        @tf.function converts the wrapped function to a static computational 
        graph. This tremendously speeds up computation but you have to write
        graph-style code inside the function. I.e. treat tensor variables as
        nodes in a flow of computation, NOT stateful objects which you
        can modify.
        Use operations of the form 
        a = tf.example(a)

        NOT 
        a.write(etc)

        You are chaining operations, not modifying states.'''
        super(Actor, self).__init__()

        self.l1 = kl.Dense(ac_layers[0], activation='relu', kernel_initializer=initialize_relu,
                          kernel_regularizer=l2(reg_coeff))
        self.l2 = kl.Dense(ac_layers[1], activation='relu', kernel_initializer=initialize_relu,
                          kernel_regularizer=l2(reg_coeff))
        self.l3 = kl.Dense(action_dim, activation='tanh', kernel_initializer=initialize_tanh,
                          kernel_regularizer=l2(reg_coeff))
        
        self._max_action = max_action
        # Remember building the weights before you can copy them
        self.build(input_shape=(None, state_dim))

    @tf.function 
    def call(self, state):
        assert state.dtype == tf.float32
        x = self.l1(state)
        x = self.l2(x)
        return self._max_action * self.l3(x)


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
        self.l4 = kl.Dense(cr_layers[0], activation='relu', kernel_initializer=initialize_relu,
                           kernel_regularizer=l2(reg_coeff))
        self.l5 = kl.Dense(cr_layers[1], activation='relu', kernel_initializer=initialize_relu,
                           kernel_regularizer=l2(reg_coeff))
        self.l6 = kl.Dense(1, kernel_regularizer=l2(reg_coeff))
        self.build(input_shape=(None, state_dim+action_dim))

    @tf.function
    def call(self, state_action):
        assert state_action.dtype == tf.float32
        q1 = self.l1(state_action) 
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
