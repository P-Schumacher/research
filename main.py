from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
from pudb import set_trace
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
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
config = {**cnf.main, **cnf.agent, **cnf.coppeliagym, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model}
if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)
main(cnf.main)
