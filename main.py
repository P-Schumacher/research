from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
import sys
import os
from pudb import set_trace
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ant_env = False
vrep = True

name = [sys.argv[1] if len(sys.argv) == 2 else None][0]
if vrep:
    main_cnf = OmegaConf.load('configs/vrep_default/vrep_main_conf.yaml')
    env_cnf = OmegaConf.load('configs/vrep_default/vrep_env_conf.yaml')
    agent_cnf = OmegaConf.load('configs/vrep_default/vrep_agent_conf.yaml')
    # Parameters of second cnf file overwrite those of first
    cnf = OmegaConf.merge(main_cnf, env_cnf, agent_cnf)
    if name:
        exp_cnf = OmegaConf.load(f'experiments/{name}.yaml')
        cnf = OmegaConf.merge(cnf, exp_cnf)

if ant_env:
    main_cnf = OmegaConf.load('configs/ant_default/ant_main_conf.yaml')
    agent_cnf = OmegaConf.load('configs/ant_default/ant_agent_conf.yaml')
    # Parameters of second cnf file overwrite those of first
    cnf = OmegaConf.merge(main_cnf, agent_cnf)
    if name:
        exp_cnf = OmegaConf.load(f'experiments/{name}.yaml')
        cnf = OmegaConf.merge(cnf, exp_cnf)
cnf.merge_with_cli()
if vrep:
    config = {**cnf.main, **cnf.agent, **cnf.coppeliagym.params, **cnf.coppeliagym.sim, **cnf.buffer,
              **cnf.agent.sub_model, **cnf.agent.meta_model}
else:
    config = {**cnf.main, **cnf.agent, **cnf.buffer, **cnf.agent.sub_model}

if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)
main(cnf)
