from per_exp.eval_grads.eval_grads import main
from per_exp.eval_grads.scatter_plot import main as main_scatter
#from per_exp.eval_grads.train_critic import main
#from per_exp.eval_grads.train_actor import main
from omegaconf import OmegaConf
import wandb
import sys
import os
from pudb import set_trace
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ant_env = False
vrep = True

if vrep:
    main_cnf = OmegaConf.load('configs/vrep_default/vrep_main_conf.yaml')
    env_cnf = OmegaConf.load('configs/vrep_default/vrep_env_conf.yaml')
    agent_cnf = OmegaConf.load('configs/vrep_default/vrep_agent_conf.yaml')
    # Parameters of second cnf file overwrite those of first
    cnf = OmegaConf.merge(main_cnf, env_cnf, agent_cnf)
    
    exp_cnf = OmegaConf.load(f'per_exp/eval_grads/constant_buffer.yaml')
    cnf = OmegaConf.merge(cnf, exp_cnf)


cnf.merge_with_cli()
if vrep:
    config = {**cnf.main, **cnf.agent, **cnf.coppeliagym.params, **cnf.coppeliagym.sim, **cnf.buffer, **cnf.agent.sub_model, **cnf.agent.meta_model}
else:
    config = {**cnf.main, **cnf.agent, **cnf.buffer, **cnf.agent.sub_model}

if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)
main_scatter(cnf)
