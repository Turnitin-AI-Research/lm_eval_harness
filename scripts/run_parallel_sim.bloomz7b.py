import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import itertools
import ray
import fire
import torch
from main import results_fpath


def run(overwrite_results):
    # DEPRICATED: DO NOT submit jobs to a cluster because jobs will be run in the environment in which the cluster was started.
    # run "ray start --head --dashboard-host 0.0.0.0" from the repo root directory from within the venv lme.
    # If you to attach another machine to the cluster, then run "ray start --address=<head-node-ip>:6379" there.
    # To view dashboard, forward local port to remote dashboard either using vscode or via ssh: ssh -L 8265:<head-node-ip>:8265 <head-node-ip>
    # ray.init(address='auto')

    # Start a new cluster in order to ensure we're using the right environment. This will prevent us from connecting to a running
    # ray cluster that was started in another environment.
    NUM_GPUS = torch.cuda.device_count()
    NUM_GPUS_PER_RUN = 2
    MAX_PARALLEL_RUNS =  NUM_GPUS // NUM_GPUS_PER_RUN
    # NUM_CPUS_PER_RUN = os.cpu_count() // MAX_PARALLEL_RUNS
    print(f'NUM_GPUS_PER_RUN={NUM_GPUS_PER_RUN}')
    ray.init(address='local', num_cpus=MAX_PARALLEL_RUNS + 8)

    results_dir = "lmeval_results_sim_bloomz71b/"
    num_fewshots = [5, 0]
    task_models = [('hellaswag_d', 'dist_sim')]  # ('hellaswag_d', 'dist_sim'), ('webqs_dg', 'dist_gen')]
    pretrained = ['bigscience/bloomz-7b1']
    # ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
    encoding_schemes = ['sentence_level_segmentation', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
    # ['-relu|mean', '-relu+|mean', 'relu+|mean', 'relu|mean', 'relu+|last', 'relu|last', '-relu+|last', 'relu+|last']
    word_agg_schemes = ['mean', 'w1mean']  # ['-relu+|mean', '-relu+|last', '-relu|last']
    segment_agg_schemes = [None]
    example_agg_schemes = [None, 'mean', 'soft_cluster']
    norms = [None, 'layer']
    sim_funcs = ['dot_product', 'cosine_sim']
    # ['middle', None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    encoding_layers = ['-2', None, 'middle', 'E', 0]  # , 'E', 0, 'middle']
    parallelize = True

    if 0 in num_fewshots:
        ALLOWED_ZEROSHOT_ENCODING_SCHEMES = {'concat_all_examples', 'segment_each_example', 'sentence_level_segmentation'}
        ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES = {None}
        assert ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES & set(example_agg_schemes)
        assert ALLOWED_ZEROSHOT_ENCODING_SCHEMES & set(encoding_schemes)

    @ray.remote(max_calls=1, num_gpus=NUM_GPUS_PER_RUN)
    # @ray.remote(max_calls=1, num_cpus=4)
    def run_eval(args):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        from main import main
        return main(*args)

    os.makedirs(results_dir, exist_ok=True)
    futures = []
    for num_fewshot, (task, model), submodel, encoding_scheme, word_agg_scheme, segment_agg_scheme, example_agg_scheme, norm, sim_func, encoding_layer in itertools.product(
        num_fewshots, task_models, pretrained, encoding_schemes, word_agg_schemes, segment_agg_schemes, example_agg_schemes, norms, sim_funcs, encoding_layers):

        if num_fewshot == 0:
            if ((encoding_scheme not in ALLOWED_ZEROSHOT_ENCODING_SCHEMES)
                    or (example_agg_scheme not in ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES)):
                continue

        _args = [
            "--device", "cpu",
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
            model_args = model_args + f',pretrained={submodel},PARALLELIZE={parallelize}'
        _args.extend(['--model_args', model_args])

        results_path = results_fpath(*_args)
        if (results_path is not None) and (not overwrite_results) and os.path.exists(results_path):
            print(f'Skipping config:\n{_args}\nFound result at {results_path}')
        else:
            print(f'Posting job:\n{_args}\nresults_path = {results_path}')
            future = run_eval.remote(_args)
            futures.append(future)

    responses = ray.get(futures)
    # for resp in responses:
    #     fpath, results = resp
    #     with open(fpath, "wt", encoding='utf-8') as f:
    #         json.dump(results, f, indent=2)

    print(responses)


def run_wrapper(shutdown_at_exit: bool = False, overwrite_results: bool = False):
    try:
        run(overwrite_results)
    except Exception as e:
        if shutdown_at_exit:
            print(e)
        else:
            raise e
    if shutdown_at_exit:
        os.system('sudo shutdown now -h')


if __name__ == '__main__':
    fire.Fire(run_wrapper)
