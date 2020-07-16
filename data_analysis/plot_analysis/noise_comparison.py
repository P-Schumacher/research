
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
        Description wrong, it returns gaussian noise that can be used to create a wiener process.
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
noise = OUNoise(0, sigma=1, ou_a=5)
for i in range(1000):

