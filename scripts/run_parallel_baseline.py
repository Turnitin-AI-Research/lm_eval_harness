import ray
import itertools
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# DEPRICATED: DO NOT submit jobs to a cluster because jobs will be run in the environment in which the cluster was started.
# run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
# If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
# To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
# ray.init(address='auto')

# Start a new cluster in order to ensure we're using the right environment. This will prevent us from connecting to a running
# ray cluster that was started in another environment.
ray.init(address='local')

results_dir = "lmeval_results_baseline/"
num_fewshots = [0, 5]
# ('hellaswag_d', 'dist_sim'), ('hellaswag', 'gpt2'), ('webqs', 'gpt2')]
task_models = [('hellaswag_dg', 'dist_gen')]  # [('hellaswag_dg', 'dist_gen'), ('hellaswag', 'gpt2'), ('webqs', 'gpt2')]
encoding_scheme = 'cross_encoding'
pretrained = ['bigscience/bloomz-7b1']
parallelize = True


@ray.remote(max_calls=1, num_gpus=8)
# @ray.remote(max_calls=1, num_cpus=4)
def run_eval(args):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from main import main
    return main(*args)

os.makedirs(results_dir, exist_ok=True)
futures = []
for num_fewshot, (task, model), submodel in itertools.product(
        num_fewshots, task_models, pretrained):
    _args = [
        "--device", "cpu" if parallelize else "0",
        "--output_dir", results_dir,
        # "--limit", "5",
        "--tasks", task,
        "--model", model,
        "--no_cache",
        '--num_fewshot', f'{num_fewshot}'
    ]
    if submodel is not None:
        _args.extend(['--model_args', f'pretrained={submodel},PARALLELIZE={parallelize}'])
    if encoding_scheme:
        _args.extend(['--task_args', f'encoding_scheme={encoding_scheme}'])
    future = run_eval.remote(_args)
    futures.append(future)

responses = ray.get(futures)
# for resp in responses:
#     fpath, results = resp
#     with open(fpath, "wt", encoding='utf-8') as f:
#         json.dump(results, f, indent=2)

print(responses)
