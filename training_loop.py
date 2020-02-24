import numpy as np
import gym
import sys
import time
import math
import tensorflow as tf
import wandb
from pudb import set_trace

from agent_files.HIRO import HierarchicalAgent
from utils.logger import Logger
from utils.utils import create_world, setup,  exponential_decay

def maybe_verbose_output(agent, action, args):
    if args.render:
        print("action: " + str(action))
        print("time is: " + str(t))
        if not args.flat_agent:
            print("goal: " + str(agent.goal))
            if agent.meta_time and args.render:
                print("GOAL POSITION: " + str(agent.goal))
                env.set_goal(agent.goal[:3])

if __name__ == "__main__":
    # Parse Arguments and create directories
    args = setup(sys.argv[1:])
    if args.log:
        wandb.init(project='exp', entity='rlpractitioner', config=args)
   
   # create objects 
    env, agent = create_world(args)
    logger = Logger()
    time_step = tf.Variable(0, dtype=tf.int64)
    
    # Load previously trained model.
    if args.load_model: agent.load_model("./models/" + str(agent.file_name))
    
    state, done = env.reset(), False
   
   # Training loop
    for t in range(int(args.max_timesteps)):
        if t < args.start_timesteps:
            action = agent.random_action(state) 
        else:
            action = agent.select_noisy_action(state)
        maybe_verbose_output(agent, action, args)
        next_state, reward, done, _ = env.step(action)
        intr_rew = agent.replay_add(state, action, reward, next_state, done)
        if t > args.start_timesteps:
            agent.train(time_step)
        state = next_state
        logger.inc(reward)
        time_step.assign_add(1)
        
        if done:
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1}+ Episode T: {logger.episode_timesteps} Reward: \
                  {logger.episode_reward}")
            # Reset environment
            state, done = env.reset(), False
            agent.reset()
            logger.log(args.log, intr_rew)
            logger.reset()
        
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate = agent.evaluation(env)
            state, done = env.reset(), False
            agent.reset()
            logger.reset(post_eval=True)
            if args.log:
                wandb.log({'eval/eval_ep_rew': avg_ep_rew, 'eval/eval_intr_rew': avg_intr_rew,
                      'eval/success_rate': success_rate})
            if args.save_model: agent.save_model("./models/"+str(agent.file_name))
