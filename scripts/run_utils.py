"""run script utils"""
from typing import Optional
import ray
import torch


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
