import os
import sys
import typing
import itertools
import json
import pandas as pd
import ray

# run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
# If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
# To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
# ray.init(address='auto')
ray.init()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

num_fewshots = [5]
task_models = [('hellaswag_d', 'dist_sim')] #, ('hellaswag_d', 'dist_sim'), ('webqs_d', 'dist_sim')]
pretrained = ['EleutherAI/gpt-neo-1.3B']
encoding_schemes = ['concat_all_examples']  # ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
word_agg_schemes = ['mean']
segment_agg_schemes = [None]
example_agg_schemes = ['soft_cluster', None, 'mean']
norms = ['layer']
sim_funcs = ['dot_product']
encoding_layers = ['middle', None]

@ray.remote(max_calls=1, num_gpus=1)
# @ray.remote(max_calls=1, num_cpus=4)
def run_eval(args):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from main import main
    return main(*args)

futures = []
for num_fewshot, (task, model), submodel, encoding_scheme, word_agg_scheme, segment_agg_scheme, example_agg_scheme, norm, sim_func, encoding_layer in itertools.product(
    num_fewshots, task_models, pretrained, encoding_schemes, word_agg_schemes, segment_agg_schemes, example_agg_schemes, norms, sim_funcs, encoding_layers):
    _args = [
        "--device", "0",
        "--output_dir", "lmeval_results_sim4/",
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

responses
