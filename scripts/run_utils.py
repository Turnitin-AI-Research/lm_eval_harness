"""run script utils"""
from typing import Optional
import ray
import pandas as pd
import torch


NUM_GPUS_BY_MODEL_SIZE = {
    750: 1,   # 1x 11GB GPU
    1500: 1,  # 1x 11GB GPU
    2500: 1,  # 1x 11GB GPU
    3500: 1,  # 1x 11GB GPU
    6500: 2,  # 2x 11GB GPU
    11500: 4,  # 4x 24GB GPUs
    13000: 4,  # 4x 24GB GPUs
    20000: 8,  # 8x 24GB GPUs
}

def ray_init(num_gpus_per_run: Optional[int] = None, cluster: str = 'local'):
    """Initialize or join ray cluster"""
    if cluster == 'local':
        # Start a new cluster in order to ensure we're using the right environment. This will prevent us from connecting to a running
        # ray cluster that was started in another environment.
        NUM_GPUS = torch.cuda.device_count()
        MAX_PARALLEL_RUNS = NUM_GPUS // (num_gpus_per_run or 1)
        print(f'num_gpus_per_run={num_gpus_per_run}')
        ray.init(address='local', num_cpus=MAX_PARALLEL_RUNS + 2)
    else:
        # run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
        # If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
        # To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
        # ray.init(address='auto')
        ray.init(address='auto')


def get_models(type: str, datadir='data', max_size: Optional[int] = None, min_size: Optional[int] = None):
    """Get models from LM_List.parquet. Prune the list by type and size. Sort by size descending."""
    lm_list_df = pd.read_parquet(f'{datadir}/LM_List.df.parquet')
    df = lm_list_df[lm_list_df.training_type.str.contains(type)]
    if max_size is not None:
        df = df[df['size'] < max_size]
    if min_size is not None:
        df = df[df['size'] >= min_size]
    return df.sort_values(by='size', ascending=False).to_dict('records')
