import tensorflow as tf
import numpy as np

def huber(dist, delta):
    return tf.reduce_sum(tf.square(delta) * ( tf.pow(1 + tf.square(dist  / delta), 0.5) - 1 ))

def euclid(dist, axis=0):
    return tf.norm(dist, axis=axis)

def huber_not_reduce(dist, delta):
    return tf.square(delta) * ( tf.pow(1 + tf.square(dist  / delta), 0.5) - 1 )

def clip_by_global_norm(t_list, clip_norm):
    '''Clips the tensors in the list of tensors *t_list* globally by their norm. This preserves the 
    relative weights of gradients if used on gradients. The inbuilt clip_norm argument of 
    keras optimizers does NOT do this. Global norm clipping is the correct way of implementing
    gradient clipping. The function *tf.clip_by_global_norm()* changes the structure of the passed tensor
    sometimes. This is why I decided not to use it.
    :param t_list: List of tensors to be clipped.
    :param clip_norm: Norm over which the tensors should be clipped.
    :return t_list: List of clipped tensors. 
    :return norm: New norm after clipping.'''
    norm = get_norm(t_list)
    if norm > clip_norm:
        t_list = [tf.scalar_mul(clip_norm / norm, t) for t in t_list]
        norm = clip_norm
    return t_list, norm

def get_norm(t_list):
    return tf.math.sqrt(sum([tf.reduce_sum(tf.square(t)) for t in t_list]))

def clip_by_global_norm_single(t, clip_norm):
    '''Clips the tensors in the list of tensors *t_list* globally by their norm. This preserves the 
    relative weights of gradients if used on gradients. The inbuilt clip_norm argument of 
    keras optimizers does NOT do this. Global norm clipping is the correct way of implementing
    gradient clipping. The function *tf.clip_by_global_norm()* changes the structure of the passed tensor
    sometimes. This is why I decided not to use it.
    :param t_list: List of tensors to be clipped.
    :param clip_norm: Norm over which the tensors should be clipped.
    :return t_list: List of clipped tensors. 
    :return norm: New norm after clipping.'''
    norm = tf.norm(t)
    if norm > clip_norm:
        t = tf.scalar_mul(clip_norm / norm, t)
        norm = clip_norm
    return t, norm

class OUNoise(object):

    def __init__(self, ou_mu, delta=0.01, sigma=10., ou_a=1.):
        # Noise parameters
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        self.first = np.zeros_like(self.ou_mu)
        self.init = False

    def brownian_motion_log_returns(self):
        """
        This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
        about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
        :return: brownian motion log returns
        """
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=self.ou_mu.shape)

    def ornstein_uhlenbeck_level(self):
        """
        This method returns the rate levels of a mean-reverting ornstein uhlenbeck process.
        :return: the Ornstein Uhlenbeck level
        """
        self.prev_ou_level = [self.prev_ou_level if self.init else self.first][0]
        drift = self.ou_a * (self.ou_mu - self.prev_ou_level) * self.delta
        randomness = self.brownian_motion_log_returns()
        self.init = True
        self.prev_ou_level = self.prev_ou_level + drift + randomness
        return self.prev_ou_level

    def vis_noise(self):
        """
        Visualize the noise to judge the parameters
        """
        a = []
        for i in range(1000):
            a.append(self.ornstein_uhlenbeck_level())
        from matplotlib import pyplot as plt
        plt.plot(a)
        plt.show()

