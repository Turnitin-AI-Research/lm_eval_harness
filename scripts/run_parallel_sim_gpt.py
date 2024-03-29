#!/usr/bin/env python
import os
import itertools
import ray
import fire
import scripts.run_utils as utils
from main import results_fpath
import logging


def run(overwrite_results: bool, NUM_GPUS_PER_RUN: int, cluster: str):
    utils.ray_init(num_gpus_per_run=NUM_GPUS_PER_RUN, cluster=cluster)
    results_dir = "lmeval_results_sim_latest/"
    num_fewshots = [5, 0]
    task_models = [('hellaswag_d', 'dist_sim')]  # ('hellaswag_d', 'dist_sim'), ('webqs_dg', 'dist_gen')]
    pretrained = utils.get_models(arch_type='Decoder Only', max_size=8000)
    parallelize: bool = True
    # ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
    encoding_schemes = ['concat_all_examples', 'concat_each_example', 'sentence_level_segmentation']
    # ['-relu|mean', '-relu+|mean', 'relu+|mean', 'relu|mean', 'relu+|last', 'relu|last', '-relu+|last', 'relu+|last']
    # ['w1mean', 'relu|w1mean', '-relu|w1mean']  # ['-relu+|mean', '-relu+|last', '-relu|last']
    word_agg_schemes = ['w1mean']
    segment_agg_schemes = ['mean']
    example_agg_schemes = ['mean']
    norms = [None]
    sim_funcs = ['dot_product']
    # ['middle', None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    encoding_layers = [None]  # ,23, 'E', 0, 'middle']
    output_enclayer_and_aggschemes: list[tuple] = [(None, None)]  # [('OE', 'mean')]
    if 0 in num_fewshots:
        ALLOWED_ZEROSHOT_ENCODING_SCHEMES = {'concat_all_examples',
                                             'segment_each_example', 'sentence_level_segmentation'}
        assert ALLOWED_ZEROSHOT_ENCODING_SCHEMES & set(encoding_schemes)

    @ray.remote(max_calls=1, num_gpus=NUM_GPUS_PER_RUN)
    # @ray.remote(max_calls=1, num_cpus=4)
    def run_eval(args):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        # os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        from main import main
        return main(*args)

    os.makedirs(results_dir, exist_ok=True)
    futures = []
    for num_fewshot, (task, model), submodel, encoding_scheme, word_agg_scheme, segment_agg_scheme, example_agg_scheme, norm, sim_func, encoding_layer, (out_encoding_layer, out_word_agg_scheme) in itertools.product(
            num_fewshots, task_models, pretrained, encoding_schemes, word_agg_schemes, segment_agg_schemes, example_agg_schemes, norms, sim_funcs, encoding_layers, output_enclayer_and_aggschemes):

        # Remove unnecessary combinations to help reign in combinatorial explosion
        if num_fewshot == 0:
            if task.startswith('hellaswag'):
                if encoding_scheme in ['concat_each_example', 'segment_each_example']:
                    encoding_scheme = 'concat_all_examples'
                if encoding_scheme == 'concat_all_examples':
                    segment_agg_scheme = None
                    example_agg_scheme = None
            if encoding_scheme not in ALLOWED_ZEROSHOT_ENCODING_SCHEMES:
                continue
        else:
            if task.startswith('hellaswag'):
                if encoding_scheme in ['segment_each_example']:
                    encoding_scheme = 'concat_each_example'
                if encoding_scheme == 'concat_all_examples':
                    segment_agg_scheme = None
                    example_agg_scheme = None
                elif encoding_scheme == 'concat_each_example':
                    segment_agg_scheme = None

        _args = [
            "--device", 0,
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
        if parallelize:
            model_args = model_args + ',PARALLELIZE=True'
        _args.extend(['--model_args', model_args])

        results_path = results_fpath(*_args)
        if (results_path is not None) and (not overwrite_results) and os.path.exists(results_path):
            logging.info(f'Skipping config:\n{_args}')
        else:
            # Call the ray remote function, passing it the number of gpus to use
            if submodel is None:
                num_gpus = NUM_GPUS_PER_RUN
            else:
                num_gpus = utils.NUM_GPUS_BY_MODEL_SIZE[submodel['model_size']]
            future = run_eval.options(num_gpus=num_gpus).remote(_args)
            # future = run_eval.remote(_args)
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
