from training_loops.training_loop import main
import argparse
from omegaconf import OmegaConf
import wandb
import sys
import os
from pudb import set_trace
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
ant_env = False
vrep = True


#name = [sys.argv[1] if len(sys.argv) >= 2 else None][0]
#seed = [sys.argv[2] if len(sys.argv) >= 3 else None][0]

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='config8')
parser.add_argument('--seed',default=0, type=int)
parser.add_argument('--alpha', default=1., type=float)
parser.add_argument('--beta', default=0., type=float)
args = parser.parse_args()
seed = args.seed
name = args.name

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
cnf.agent.meta_model.alpha = args.alpha
cnf.agent.sub_model.beta = args.beta
if seed:
    cnf.main.seed = int(seed)

if vrep:
    config = {**cnf.main, **cnf.agent, **cnf.coppeliagym.params, **cnf.coppeliagym.sim, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model}
else:
    config = {**cnf.main, **cnf.agent, **cnf.buffer, **cnf.agent.sub_model}

if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)
main(cnf)
