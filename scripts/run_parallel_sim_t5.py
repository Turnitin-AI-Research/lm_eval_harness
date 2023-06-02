#!/usr/bin/env python
import os
import fire
import scripts.run_utils as utils
# from main import results_fpath


def run(overwrite_results: bool, NUM_GPUS_PER_RUN: int, cluster: str):
    results_dir = "lmeval_results_t5/"
    num_fewshots = [5, 0]
    task_models = [('hellaswag_d', 'dist_sim')]  # ('hellaswag_d', 'dist_sim'), ('webqs_dg', 'dist_gen')]
    # ['google/flan-t5-xl']  # ['T0pp', 'flan-t5']
    pretrained = utils.get_models(arch_type='Encoder-Decoder', max_size=13000)
    parallelize: bool = True
    # ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
    encoding_schemes = ['sentence_level_segmentation',
                        'segment_each_example', 'concat_each_example', 'concat_all_examples']
    # ['-relu|mean', '-relu+|mean', 'relu+|mean', 'relu|mean', 'relu+|last', 'relu|last', '-relu+|last', 'relu+|last']
    # ['w1mean', 'relu|w1mean', '-relu|w1mean']  # ['-relu+|mean', '-relu+|last', '-relu|last']
    word_agg_schemes = ['mean']
    segment_agg_schemes = ['mean', None]
    example_agg_schemes = [None, 'mean']
    norms = [None]
    sim_funcs = ['dot_product']
    # ['middle', None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    encoding_layers = [None]  # ,23, 'E', 0, 'middle']
    output_enclayer_and_aggschemes: list[tuple] = [(None, None)]  # [('OE', 'mean')]

    responses = utils.run_parallel(
        results_dir=results_dir,
        overwrite_results=overwrite_results,
        cluster=cluster,
        num_fewshots=num_fewshots,
        task_models=task_models,
        pretrained=pretrained,
        encoding_schemes=encoding_schemes,
        word_agg_schemes=word_agg_schemes,
        segment_agg_schemes=segment_agg_schemes,
        example_agg_schemes=example_agg_schemes,
        norms=norms,
        sim_funcs=sim_funcs,
        encoding_layers=encoding_layers,
        output_enclayer_and_aggschemes=output_enclayer_and_aggschemes,
        NUM_GPUS_PER_RUN=NUM_GPUS_PER_RUN,
        parallelize=parallelize
    )
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
