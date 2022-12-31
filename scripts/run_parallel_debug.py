import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import itertools
import ray


# DEPRICATED: DO NOT submit jobs to a cluster because jobs will be run in the environment in which the cluster was started.
# run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
# If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
# To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
# ray.init(address='auto')

# Start a new cluster in order to ensure we're using the right environment. This will prevent us from connecting to a running
# ray cluster that was started in another environment.
ray.init(address='local')

results_dir = "lmeval_results_debug/"
num_fewshots = [0, 5]
# [('hellaswag_dg', 'dist_gen'), ('hellaswag_d', 'dist_sim'), ('hellaswag', 'gpt2'), ('webqs', 'gpt2')]
task_models = [('webqs_dg', 'dist_gen')]
pretrained = ['EleutherAI/gpt-neo-1.3B']
# ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
encoding_schemes = ['cross_encoding']
# ['-relu|mean', '-relu+|mean', 'relu+|mean', 'relu|mean', 'relu+|last', 'relu|last', '-relu+|last', 'relu+|last']
word_agg_schemes = [None]
segment_agg_schemes = [None]
example_agg_schemes = [None]  # [None, 'mean', 'soft_cluster']
norms = [None]
sim_funcs = [None]
encoding_layers = [None]  # ['middle', None, 23, 'E', 0]  # ['middle', None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

@ray.remote(max_calls=1, num_gpus=1)
# @ray.remote(max_calls=1, num_cpus=4)
def run_eval(args):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from main import main
    return main(*args)

os.makedirs(results_dir, exist_ok=True)
futures = []
for num_fewshot, (task, model), submodel, encoding_scheme, word_agg_scheme, segment_agg_scheme, example_agg_scheme, norm, sim_func, encoding_layer in itertools.product(
    num_fewshots, task_models, pretrained, encoding_schemes, word_agg_schemes, segment_agg_schemes, example_agg_schemes, norms, sim_funcs, encoding_layers):
    _args = [
        "--device", "0",
        "--output_dir", results_dir,
        # "--limit", "5",
        "--tasks", task,
        "--model", model,
        "--no_cache",
        '--num_fewshot', f'{num_fewshot}',
        '--task_args', f'encoding_scheme={encoding_scheme}'
    ]
    model_args = f'WORD_AGG_SCHEME={word_agg_scheme},EXAMPLE_AGG_SCHEME={example_agg_scheme},SEGMENT_AGG_SCHEME={segment_agg_scheme},NORM={norm},SIMILARITY_FUNC={sim_func},ENCODING_LAYER={encoding_layer}'
    if submodel is not None:
        model_args = model_args + f',pretrained={submodel}'
    _args.extend(['--model_args', model_args])
    future = run_eval.remote(_args)
    futures.append(future)

responses = ray.get(futures)
# for resp in responses:
#     fpath, results = resp
#     with open(fpath, "wt", encoding='utf-8') as f:
#         json.dump(results, f, indent=2)

print(responses)
