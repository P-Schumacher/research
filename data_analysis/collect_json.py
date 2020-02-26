import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import wandb
import sys
from pudb import set_trace
api = wandb.Api()

project = sys.argv[1]
runs = api.runs('rlpractitioner/'+project)

keys = ['_step', 'eval/eval_ep_rew', 'eval/eval_intr_rew', 'eval/success_rate', 'c_step']

name_list = []
config_list = []
for run in runs:
    name_list.append(run.name)
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})
df = pd.DataFrame.from_records({'name': name_list})

rows = []
for key_idx, key in enumerate(keys):
    print(f"{key_idx} of {len(keys)}")
    columns = []
    for run_idx, run in enumerate(runs):
        data = run.history(samples=10000, keys=[key])
        if key == '_step':
            columns.append(np.array(run.history()['_step']))
        else:
            columns.append(np.array([data[x][0][key] for x in data], dtype=np.float64))
        
    concat_df = pd.DataFrame.from_records({key: columns})
    df = pd.concat([df, concat_df], axis=1)    

df.to_json(project+'.json')
