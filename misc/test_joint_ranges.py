import numpy as np
import tensorflow as tf
from environments.coppeliagym import CoppeliaEnv    
from utils.utils import setup, create_world
import collections

args = ['--ee_j_pos', '--vrep', '--render']
args = setup(args)
env, agent = create_world(args)

for eps in range(10):
    state = env.reset()
    env.render()
    for t in range(args.time_limit):
        if t < 150:  
            action = env.action_space.high
        else:
            action = env.action_space.low
        np.putmask(action, [1,1,1,1,0,1,1,1], 0)
        next_state, *_ = env.step(action)
        print(next_state[7])




    
