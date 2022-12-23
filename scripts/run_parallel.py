"""Run eval using ray. Run it from the repo root dir"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import ray
import pandas as pd
import json
from pathlib import Path
import itertools


# run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
# If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
# To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
ray.init(address='auto')


num_fewshots = [5]
# , ('hellaswag_d', 'dist_sim'), ('webqs_dg', 'dist_gen')]
task_models = [('hellaswag_d', 'dist_sim')]  # [('hellaswag_dg', 'dist_gen'), ('webqs_dg', 'dist_gen')]
pretrained = ['EleutherAI/gpt-neo-1.3B']
# ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
encoding_schemes = ['concat_all_examples']
word_agg_schemes = ['mean']
segment_agg_schemes = [None]
example_agg_schemes = ['soft_cluster', 'mean', None]  # ['mean', None, 'soft_cluster']
norms = ['layer']
sim_funcs = ['dot_product']
encoding_layers = ['middle', None]  # ['E', 'average:0-4', 'average:5-9', 'average:10-14', 'average:15-19', 'average:20-24']  # middle, average:<start num>-<end num>, <num>
steering_params = [  # (DECODING_SCHEME, STEER_VEC_INJ_LAYERS, STEER_VEC_INJ_POS)
                    (None, None, None),
                    # ('steer_vec', 'all', 'all'), ('steer_vec', 'all', '0'),
                    # ('steer_vec', '8-15', 'all'), ('steer_vec', '8-15', '0'),
                    # ('steer_vec', '10-12', 'all'), ('steer_vec', '10-12', '0'),
                    # ('steer_vec', '20-24', 'all'), ('steer_vec', '20-24', '0')
                    ]


@ray.remote(max_calls=1, num_gpus=1)
# @ray.remote(max_calls=1, num_cpus=4)
def run_eval(args):
    import os as _os
    _os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from main import main
    print(f'Running ray remote func with args: {args}')
    return main(*args)


futures = []
for num_fewshot, (task, model), submodel, encoding_scheme, word_agg_scheme, segment_agg_scheme, example_agg_scheme, norm, sim_func, encoding_layer, (decoding_scheme, inj_layers, inj_pos) in itertools.product(
        num_fewshots, task_models, pretrained, encoding_schemes, word_agg_schemes, segment_agg_schemes, example_agg_schemes, norms, sim_funcs, encoding_layers, steering_params):
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
    model_args = (f'WORD_AGG_SCHEME={word_agg_scheme},EXAMPLE_AGG_SCHEME={example_agg_scheme},'
                  + f'SEGMENT_AGG_SCHEME={segment_agg_scheme},NORM={norm},SIMILARITY_FUNC={sim_func},ENCODING_LAYER={encoding_layer}')
    if decoding_scheme is not None:
        model_args += f',DECODING_SCHEME={decoding_scheme},STEER_VEC_INJ_LAYERS={inj_layers},STEER_VEC_INJ_POS={inj_pos}'
    if submodel is not None:
        model_args = model_args + f',pretrained={submodel}'
    _args.extend(['--model_args', model_args])
    future = run_eval.remote(_args)
    futures.append(future)

responses = ray.get(futures)
for resp in responses:
    fpath, results = resp
    with open(fpath, "wt", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
