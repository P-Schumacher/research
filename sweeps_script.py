import sys
from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
from pudb import set_trace
import numpy as np
import argparse

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
cnf.main.log = 1
cnf.main.eval_freq = 20000
cnf.project = 'param_sweep'
cnf.coppeliagym.params.force = 0
config = {**cnf.main, **cnf.agent, **cnf.coppeliagym, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model, **cnf.coppeliagym.params}

wandb.init(project=cnf.project, entity=cnf.entity, config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--sub_model_ac_lr', default=0.001, type=float)
parser.add_argument('--sub_model_cr_lr', default=0.0001, type=float)
parser.add_argument('--sub_model_reg_coeff_ac', default=0.0, type=float)
parser.add_argument('--sub_model_reg_coeff_cr', default=0.0, type=float)
parser.add_argument('--sub_model_clip_ac', default=10000.0, type=float)
parser.add_argument('--sub_model_clip_cr', default=10000.0, type=float)
parser.add_argument('--action_regularizer', default=0.0, type=float)
parser.add_argument('--start_timesteps', default=1000.0, type=int)
parser.add_argument('--sub_noise', default=0.5, type=float)
parser.add_argument('--sub_rew_scale', default=1., type=float)
parser.add_argument('--max_size', default=200000, type=int)

args = parser.parse_args(sys.argv[1:])
cnf.agent.sub_model.ac_lr = args.sub_model_ac_lr
cnf.agent.sub_model.cr_lr = args.sub_model_cr_lr
cnf.agent.sub_model.reg_coeff_ac = args.sub_model_reg_coeff_ac
cnf.agent.sub_model.reg_coeff_cr = args.sub_model_reg_coeff_cr
cnf.agent.sub_model.clip_ac = args.sub_model_clip_ac
cnf.agent.sub_model.clip_cr = args.sub_model_clip_cr
cnf.coppeliagym.params.action_regularizer = args.action_regularizer
cnf.main.start_timesteps = args.start_timesteps
cnf.agent.sub_noise = args.sub_noise
cnf.agent.sub_rew_scale = args.sub_rew_scale
cnf.buffer.max_size = args.max_size

main(cnf)




