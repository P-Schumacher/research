from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
from pudb import set_trace
import sys
import os
from utils.utils import set_seeds
from mpi4py import MPI
import numpy as np
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

cnf.main.max_timesteps = 200000
cnf.main.log = 1
cnf.main.eval_freq = 20000
cnf.project = 'param_sweep'

#___________RANGES______________________

# sub 
ac_lr = np.random.uniform(0.01, 0.0001)
cr_lr = np.random.uniform(0.01, 0.0001)
reg_coeff_ac = np.random.uniform(0, 1e-2)
reg_coeff_cr= np.random.uniform(0, 1e-2)
clip_ac = np.random.uniform(0.1, 10)
clip_cr = np.random.uniform(0.1, 10)
# env 
force = np.random.choice(np.array([0,1], dtype=np.int32))
#force = 0
action_regularizer = np.random.uniform(0, 1e-3)
# main
gradient_steps = np.random.choice([1, 300])
#gradient_steps = 300 
start_timesteps = int(np.random.uniform(0, 10000))
# agent
if force:
    sub_noise = np.random.uniform(0, 200)
else:
    sub_noise = np.random.uniform(0, 3)
sub_rew_scale = np.random.uniform(0.1, 10)
# buffer
max_size = np.random.uniform(20000, 1000000)
#______________ VALUES ____________________
# sub
cnf.agent.sub_model.ac_lr = ac_lr
cnf.agent.sub_model.cr_lr = cr_lr
cnf.agent.sub_model.reg_coeff_ac = reg_coeff_ac
cnf.agent.sub_model.reg_coeff_cr = reg_coeff_cr
cnf.agent.sub_model.clip_ac = clip_ac
cnf.agent.sub_model.clip_cr = clip_cr
# env
cnf.coppeliagym.params.force = int(force)
cnf.coppeliagym.params.action_regularizer = action_regularizer
# main
cnf.main.gradient_steps = int(gradient_steps)
cnf.main.train_every = int(gradient_steps)
cnf.main.start_timesteps = start_timesteps
#buffer
cnf.buffer.max_size = int(max_size)

config = {**cnf.main, **cnf.agent, **cnf.coppeliagym, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model}
if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)


print(main(cnf))
