"""run script utils"""
import ray
import torch


def ray_init(num_gpus_per_run: int):
    """Initialize or join ray cluster"""
    # DEPRICATED: DO NOT submit jobs to a cluster because jobs will be run in the environment in which the cluster was started.
    # run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
    # If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
    # To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
    # ray.init(address='auto')

    # Start a new cluster in order to ensure we're using the right environment. This will prevent us from connecting to a running
    # ray cluster that was started in another environment.
    NUM_GPUS = torch.cuda.device_count()
    MAX_PARALLEL_RUNS =  NUM_GPUS // num_gpus_per_run
    print(f'num_gpus_per_run={num_gpus_per_run}')
    ray.init(address='local', num_cpus=MAX_PARALLEL_RUNS + 2)
