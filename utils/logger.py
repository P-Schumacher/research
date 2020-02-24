import wandb

class Logger:
    def __init__(self):
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0

    def inc(self, reward):
        self.episode_timesteps += 1
        self.episode_reward += reward

    def reset(self, post_eval=False):
        self.episode_timesteps = 0
        self.episode_reward = 0
        if not post_eval:
            self.episode_num += 1
    
    def log(self, logging, intr_rew):
        if logging:
            wandb.log({'ep_rew': self.episode_reward, 'intr_reward': intr_rew})
