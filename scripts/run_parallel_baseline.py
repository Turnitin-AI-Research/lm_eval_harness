#!/usr/bin/env python
import os
import itertools
import ray
import fire
import scripts.run_utils as utils
from main import results_fpath


def run(overwrite_results: bool, NUM_GPUS_PER_RUN: int, cluster: str):
    utils.ray_init(num_gpus_per_run=NUM_GPUS_PER_RUN, cluster=cluster)

    results_dir = "lmeval_results_baseline/"
    num_fewshots = [0, 5]
    # ('hellaswag_d', 'dist_sim'), ('hellaswag', 'gpt2'), ('webqs', 'gpt2')]
    # [('hellaswag_dg', 'dist_gen'), ('hellaswag', 'gpt2'), ('webqs', 'gpt2')]
    task_models = [('hellaswag_dg', 'dist_gen')]
    encoding_scheme = 'cross_encoding'
    pretrained = ['EleutherAI/gpt-neo-2.7B']  # EleutherAI/gpt-j-6B, EleutherAI/gpt-neo-1.3B, bigscience/bloomz-7b1
    parallelize = True

    @ray.remote(max_calls=1, num_gpus=NUM_GPUS_PER_RUN)
    def run_eval(args):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        # os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        from main import main
        return main(*args)

    os.makedirs(results_dir, exist_ok=True)
    futures = []
    for num_fewshot, (task, model), submodel in itertools.product(
            num_fewshots, task_models, pretrained):
        _args = [
            "--device", "0",
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

        results_path = results_fpath(*_args)
        if (results_path is not None) and (not overwrite_results) and os.path.exists(results_path):
            print(f'Skipping config:\n{_args}')
        else:
            future = run_eval.remote(_args)
            futures.append(future)

    responses = ray.get(futures)
    # for resp in responses:
    #     fpath, results = resp
    #     with open(fpath, "wt", encoding='utf-8') as f:
    #         json.dump(results, f, indent=2)

    print(responses)
    return responses


def run_wrapper(shutdown_at_exit: bool = False, overwrite_results: bool = False, NUM_GPUS_PER_RUN: int = 1, cluster: str = 'auto'):
    try:
        run(overwrite_results=overwrite_results, NUM_GPUS_PER_RUN=NUM_GPUS_PER_RUN, cluster=cluster)
    except Exception as e:
        if shutdown_at_exit:
            print(e)
        else:
            raise e
    if shutdown_at_exit:
        os.system('sudo shutdown now -h')


if __name__ == '__main__':
    fire.Fire(run_wrapper)
