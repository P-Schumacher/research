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
from utils.utils import create_world, exponential_decay

def maybe_verbose_output(t, agent, env, action, args):
    if args.render:
        print("action: " + str(action))
        print("time is: " + str(t))
        if not args.flat_agent:
            print("goal: " + str(agent.goal))
            if agent.meta_time and args.render:
                print("GOAL POSITION: " + str(agent.goal))
                env.set_goal(agent.goal[:3])

def decay_step(decay, stepper, agent):
    c_step = agent.args.c_step
    if decay:
        c_step = int(next(stepper))
        agent.goal_every_n = c_step
        agent.c_step = c_step
        agent.meta_agent.c_step = c_step
    return c_step


def main(args):
    if args.log:
        wandb.init(project='exp', entity='rlpractitioner', config=args)
    # create objects 
    env, agent = create_world(args)
    logger = Logger()
    stepper = exponential_decay(**args.step_decayer)
    time_step = tf.Variable(0, dtype=tf.int64)
    
    # Load previously trained model.
    if args.load_model: agent.load_model("./models/" + str(agent.file_name))

    # Training loop
    state, done = env.reset(), False
    for t in range(int(args.max_timesteps)):
        c_step = decay_step(args.decay, stepper, agent)
        if t < args.start_timesteps:
            action = agent.random_action(state) 
        else:
            action = agent.select_noisy_action(state)
        maybe_verbose_output(t, agent, env, action, args)
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
            logger.log(args.log, intr_rew, c_step)
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
