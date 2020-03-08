from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
from pudb import set_trace
import sys
import os
from utils.utils import set_seeds
from mpi4py import MPI
import numpy as np
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ant_env = False
vrep = True


default_cnf = OmegaConf.load('configs/default_conf.yaml')

if vrep:
    main_cnf = OmegaConf.load('configs/vrep/vrep_main_conf.yaml')
    env_cnf = OmegaConf.load('configs/vrep/vrep_env_conf.yaml')

if ant_env:
    cnf = OmegaConf.load('configs/ant_conf.yaml')
# Parameters of second cnf file overwrite those of first
cnf = OmegaConf.merge(default_cnf, main_cnf, env_cnf)
cnf.merge_with_cli()

cnf.main.max_timesteps = 400000
cnf.main.log = 0
cnf.main.eval_freq = 20000
cnf.project = 'param_sweep'


# sub 
cnf.agent.sub_model.ac_lr = np.random.uniform(0.001, 0.0001)
cnf.agent.sub_model.cr_lr = np.random.uniform(0.001, 0.0001)
cnf.agent.sub_model.reg_coeff_ac = np.random.uniform(0, 1e-4)
cnf.agent.sub_model.reg_coeff_cr= np.random.uniform(0, 1e-4)
cnf.agent.sub_model.clip_ac = np.random.uniform(0.1, 5)
cnf.agent.sub_model.clip_cr = np.random.uniform(0.1, 5)
# env 
cnf.coppeliagym.params.force = int(np.random.choice(np.array([0,1], dtype=np.int32)))
cnf.coppeliagym.params.action_regularizer = np.random.uniform(0, 1e-3)
# main
cnf.main.gradient_steps = int( np.random.choice([300]))
cnf.main.train_every = cnf.main.gradient_steps
cnf.main.start_timesteps = int(np.random.uniform(1000, 10000))
# agent
if cnf.coppeliagym.params.force:
    cnf.agent.sub_noise = np.random.uniform(0, 200)
else:
    cnf.agent.sub_noise = np.random.uniform(0, 3)
cnf.agent.sub_rew_scale = np.random.uniform(0.1, 7)
# buffer
cnf.buffer.max_size = int(np.random.uniform(20000, 1000000))

config = {**cnf.main, **cnf.agent, **cnf.coppeliagym, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model, **cnf.coppeliagym.params}
if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)

ret = main(cnf)
f = open(f'./param_scores/output_{int(time.time() % 60)}_{rank}.txt', 'w')
f.write(f'The score is: {ret}, params are: {cnf.pretty()}')
