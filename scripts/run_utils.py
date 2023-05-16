"""run script utils"""
from typing import Optional
import os
import itertools
import ray
import pandas as pd
import torch
import scripts.run_utils as utils
from main import results_fpath

NUM_DEC_GPUS_BY_MODEL_SIZE = {
    750: 1,   # 1x 11GB GPU
    1500: 1,  # 1x 11GB GPU
    2500: 2,  # 2x 11GB GPU
    3500: 2,  # 2x 11GB GPU
    6500: 3,  # 3x 11GB GPU
    11500: 6,  # 6x 11GB GPUs
    13000: 7,  # 7x 11GB GPUs
    20000: 8,  # 8x 24GB GPUs
}
NUM_GPUS_BY_MODEL_SIZE = NUM_DEC_GPUS_BY_MODEL_SIZE
NUM_ENCDEC_GPUS_BY_MODEL_SIZE = {
    750: 1,   # 1x 11GB GPU
    1500: 1,  # 1x 11GB GPU
    2500: 1,  # 1x 11GB GPU
    3500: 2,  # 2x 11GB GPU
    6500: 2,  # 3x 11GB GPU
    11500: 4,  # 6x 11GB GPUs
    13000: 4,  # 7x 11GB GPUs
    20000: 4,  # 8x 24GB GPUs
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


def get_models(*,
               arch_type: Optional[str] = None,
               training_type: Optional[str] = None,
               datadir='data',
               max_size: Optional[int] = None,
               min_size: Optional[int] = None):
    """Get models from LM_List.parquet. Prune the list by type and size. Sort by size descending."""
    df = pd.read_parquet(f'{datadir}/LM_List.df.parquet')
    if arch_type is not None:
        df = df[df.training_type.str.contains(arch_type)]
    if training_type is not None:
        df = df[df.training_type.str.contains(training_type)]
    if max_size is not None:
        df = df[df['size'] < max_size]
    if min_size is not None:
        df = df[df['size'] >= min_size]
    return df.sort_values(by='size', ascending=False).to_dict('records')


def num_gpus_by_model(model_desc: dict):
    """Get number of GPUs required for a model"""
    if 'Decoder Only' in model_desc['training_type']:
        return NUM_DEC_GPUS_BY_MODEL_SIZE[model_desc['size']]
    else:
        return NUM_ENCDEC_GPUS_BY_MODEL_SIZE[model_desc['size']]


def run_parallel(*,
                 results_dir,
                 overwrite_results,
                 cluster,
                 num_fewshots,
                 task_models,
                 pretrained,
                 encoding_schemes,
                 word_agg_schemes,
                 segment_agg_schemes,
                 example_agg_schemes,
                 norms,
                 sim_funcs,
                 encoding_layers,
                 output_enclayer_and_aggschemes,
                 NUM_GPUS_PER_RUN,
                 parallelize,
                 device=0
                 ):

    utils.ray_init(cluster=cluster)

    if 0 in num_fewshots:
        ALLOWED_ZEROSHOT_ENCODING_SCHEMES = {'concat_all_examples',
                                             'segment_each_example', 'sentence_level_segmentation'}
        ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES = {None}
        assert ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES & set(example_agg_schemes)
        assert ALLOWED_ZEROSHOT_ENCODING_SCHEMES & set(encoding_schemes)

    @ray.remote(max_calls=1, num_gpus=NUM_GPUS_PER_RUN)
    # @ray.remote(max_calls=1, num_cpus=4)
    def run_eval(args):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        from main import main
        return main(*args)

    os.makedirs(results_dir, exist_ok=True)
    futures = []
    for num_fewshot, (task, model), submodel, encoding_scheme, word_agg_scheme, segment_agg_scheme, example_agg_scheme, norm, sim_func, encoding_layer, (out_encoding_layer, out_word_agg_scheme) in itertools.product(
            num_fewshots, task_models, pretrained, encoding_schemes, word_agg_schemes, segment_agg_schemes, example_agg_schemes, norms, sim_funcs, encoding_layers, output_enclayer_and_aggschemes):

        if num_fewshot == 0:
            if ((encoding_scheme not in ALLOWED_ZEROSHOT_ENCODING_SCHEMES)
                    or (example_agg_scheme not in ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES)):
                continue

        if submodel is None:
            num_gpus = NUM_GPUS_PER_RUN
        else:
            num_gpus = utils.num_gpus_by_model(submodel)

        _args = [
            "--device", device,
            "--output_dir", results_dir,
            # "--limit", "5",
            "--tasks", task,
            "--model", model,
            "--no_cache",
            '--num_fewshot', f'{num_fewshot}',
            '--task_args', f'encoding_scheme={encoding_scheme}'
        ]
        model_args = (f'WORD_AGG_SCHEME={word_agg_scheme},EXAMPLE_AGG_SCHEME={example_agg_scheme}'
                      + f',SEGMENT_AGG_SCHEME={segment_agg_scheme},NORM={norm},SIMILARITY_FUNC={sim_func},ENCODING_LAYER={encoding_layer}')
        if submodel is not None:
            model_args = model_args + f',pretrained={submodel["model_name"]}'
        if out_word_agg_scheme is not None:
            model_args = model_args + f',OUT_WORD_AGG_SCHEME={out_word_agg_scheme}'
        if out_encoding_layer is not None:
            model_args = model_args + f',OUT_ENCODING_LAYER={out_encoding_layer}'
        if parallelize and num_gpus > 1:
            model_args = model_args + ',PARALLELIZE=True'
        _args.extend(['--model_args', model_args])

        results_path = results_fpath(*_args)
        if (results_path is not None) and (not overwrite_results) and os.path.exists(results_path):
            print(f'Skipping config:\n{_args}')
        else:
            # Call the ray remote function, passing it the number of gpus to use
            future = run_eval.options(num_gpus=num_gpus).remote(_args)
            # future = run_eval.remote(_args)
            futures.append(future)

    responses = ray.get(futures)
    # for resp in responses:
    #     fpath, results = resp
    #     with open(fpath, "wt", encoding='utf-8') as f:
    #         json.dump(results, f, indent=2)
    return responses
