import wandb
from pudb import set_trace

class Logger:
    def __init__(self, log, minilog, time_limit):
        '''Helper class that logs variables using wandb.
        It counts the episode_reward, the timesteps it took and the 
        number of episodes.
        :param log: Should we log things at all.
        :return: None'''
        self.minilog = minilog
        self.logging = log
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0

    def inc(self, t, reward):
        '''Logs variables that should be logged every timestep, not 
        just at the env of every episode.
        :param reward: The one timestep reward to be logged.
        :param t: The current timestep. We do not keep track of it inside
        the logger to ensure consistency across files.
        :return: None'''
        self.episode_timesteps += 1
        self.episode_reward += reward
        if self.logging and not self.minilog:
            wandb.log({'step_rew': reward}, step = t)

    def reset(self, post_eval=False):
        '''Resets the logging values.
        :param post_eval: If True, increments the episode_num
        :return: None'''
        self.episode_timesteps = 0
        self.episode_reward = 0
        if not post_eval:
            self.episode_num += 1
    
    def log(self, t, intr_rew, c_step):
        '''Call this function at the end of an episode. The function arguments are logged by argument passage, the
        episode_reward is tracked internally.
        :param intr_rew: The episode intrinsic reward of the sub-agent.
        :param c_step: The current number of timesteps between subgoals.
        :return: None'''
        if self.logging:
            wandb.log({'ep_rew': self.episode_reward}, step=t)
            if not self.minilog:
                wandb.log({'intr_reward': intr_rew, 'c_step': c_step}, step = t)

    def log_eval(self, t, eval_rew, eval_intr_rew, eval_success, rate_correct_solves, untouchable_steps):
        '''Log the evaluation metrics.
        :param t: The current timestep.
        :param eval_rew: The average episode reward of the evaluative episods.
        :param eval_intr_rew: The average intrinsic episode reward of the evaluative episodes.
        :param eval_success: The average success rate of the evaluative episodes.
        :return: None'''
        if self.logging:
            wandb.log({'eval/eval_ep_rew': eval_rew, 'eval/eval_intr_rew': eval_intr_rew,
                       'eval/success_rate': eval_success, 'eval/rate_correct_solves':rate_correct_solves,
                       'eval/untouchable_steps': untouchable_steps}, step = t)

    def most_important_plot(self, agent, state, action, reward, next_state, done):
        '''Log the current critic estimate and the learning target separately.
        According to maitre Wilmot this is the most important diagnostic plot in RL'''
        if self.logging and not self.minilog:
            #current_estimate_sub, learning_target_sub = agent._sub_agent.get_current_estimate_and_learning_target(state, action, reward, next_state, done)
            goal = agent.goal
            reward *= 0.1
            current_estimate_meta, learning_target_meta = agent._meta_agent.get_current_estimate_and_learning_target(state, goal, reward, next_state, done)
            #wandb.logging({'sub/current_estimate': current_estimate_sub, 'sub/learning_target': learning_target_sub}, commit=False)
            if not self.minilog:
                wandb.log({'meta/current_estimate': current_estimate_meta, 'meta/learning_target': learning_target_meta}, commit=False)
