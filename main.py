from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
from pudb import set_trace
import sys

ant_env = False
vrep = True
show = False


default_cnf = OmegaConf.load('configs/default_conf.yaml')
if vrep:
    cnf = OmegaConf.load('configs/vrep_conf.yaml')
if ant_env:
    cnf = OmegaConf.load('configs/ant_conf.yaml')
# Parameters of second cnf file overwrite those of first
cnf = OmegaConf.merge(default_cnf, cnf)
cnf.merge_with_cli()
OmegaConf.set_struct(cnf, True)

wandb.init(project=cnf.project, entity=cnf.entity, config=cnf)
main(cnf)
