import sys
from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
from pudb import set_trace
import numpy as np
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ant_env = False
vrep = True



if vrep:
    main_cnf = OmegaConf.load('configs/vrep_default/vrep_main_conf.yaml')
    env_cnf = OmegaConf.load('configs/vrep_default/vrep_env_conf.yaml')
    agent_cnf = OmegaConf.load('configs/vrep_default/vrep_agent_conf.yaml')

if ant_env:
    cnf = OmegaConf.load('configs/ant_conf.yaml')
# Parameters of second cnf file overwrite those of first
cnf = OmegaConf.merge(main_cnf, env_cnf, agent_cnf)
cnf.merge_with_cli()

cnf.main.max_timesteps = 100000
cnf.main.log = 1
cnf.main.eval_freq = 20000
cnf.project = 'clip_sweep'
cnf.coppeliagym.params.force = 0
config = {**cnf.main, **cnf.agent, **cnf.coppeliagym, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model, **cnf.coppeliagym.params}


parser = argparse.ArgumentParser()
for name, model in zip(['sub', 'meta'], [cnf.agent.sub_model, cnf.agent.meta_model]):
    parser.add_argument(f'--{name}_model_ac_lr', default=model.ac_lr, type=float)
    parser.add_argument(f'--{name}_model_cr_lr', default=model.ac_lr, type=float)
    parser.add_argument(f'--{name}_model_reg_coeff_ac', default=model.reg_coeff_ac, type=float)
    parser.add_argument(f'--{name}_model_reg_coeff_cr', default=model.reg_coeff_cr, type=float)
    parser.add_argument(f'--{name}_model_clip_ac', default=model.clip_ac, type=float)
    parser.add_argument(f'--{name}_model_clip_cr', default=model.clip_cr, type=float)
    parser.add_argument(f'--{name}_model_policy_freq', default=model.policy_freq, type=int)
    parser.add_argument(f'--{name}_model_policy_noise', default=model.policy_noise, type=float)
    parser.add_argument(f'--{name}_model_noise_clip', default=model.noise_clip, type=float)
    parser.add_argument(f'--{name}_model_tau', default=model.tau, type=float)
    parser.add_argument(f'--{name}_model_discount', default=model.discount, type=float)

parser.add_argument('--action_regularizer', default=cnf.coppeliagym.params.action_regularizer, type=float)
parser.add_argument('--start_timesteps', default=cnf.main.start_timesteps, type=int)
parser.add_argument('--sub_noise', default=cnf.agent.sub_noise, type=float)
parser.add_argument('--sub_rew_scale', default=cnf.agent.sub_rew_scale, type=float)
parser.add_argument('--max_size', default=cnf.buffer.max_size, type=int)

args = parser.parse_args(sys.argv[1:])
for name, model in zip(['sub_model', 'meta_model'], [cnf.agent.sub_model, cnf.agent.meta_model]):   
    model.ac_lr = getattr(args, f'{name}_ac_lr')
    model.cr_lr = getattr(args, f'{name}_cr_lr')
    model.reg_coeff_ac = getattr(args, f'{name}_reg_coeff_ac')
    model.reg_coeff_cr = getattr(args, f'{name}_reg_coeff_cr')
    model.clip_ac = getattr(args, f'{name}_clip_ac')
    model.clip_cr = getattr(args, f'{name}_clip_cr')
    model.policy_noise = getattr(args, f'{name}_policy_noise')
    model.policy_freq = getattr(args, f'{name}_policy_freq')
    model.noise_clip = getattr(args, f'{name}_noise_clip')
    model.tau= getattr(args, f'{name}_tau')
    model.discount = getattr(args, f'{name}_discount')
cnf.coppeliagym.params.action_regularizer = args.action_regularizer
cnf.main.start_timesteps = args.start_timesteps
cnf.agent.sub_noise = args.sub_noise
cnf.agent.sub_rew_scale = args.sub_rew_scale
cnf.buffer.max_size = args.max_size

wandb.init(project=cnf.project, entity=cnf.entity, config=config)
main(cnf)




