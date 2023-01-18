from typing import Optional, Union
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import fire
from main import main as run_eval


def _test(limit: int = None, device: str = "7", model_parallel: bool = False):
    if model_parallel:
        device = 'cpu'
        print('Setting device to "cpu" for model-parallel mode')
    _args = [
        "--device", str(device),
        "--output_dir", "lmeval_results_debug/",
        "--tasks", 'hellaswag_d',
        "--model", 'dist_sim',
        "--no_cache",
        '--num_fewshot', '0',
        '--task_args', 'encoding_scheme=sentence_level_segmentation',  # merge_all_segments
        '--model_args', ('WORD_AGG_SCHEME=relu|mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B'
                         + ',ENCODING_LAYER=23'
                         #  + f',PARALLELIZE={model_parallel}'
                         #  + ',DECODING_SCHEME=steer_vec,STEER_VEC_INJ_LAYERS=10-12,STEER_VEC_INJ_POS=all'
                         )
    ]
    if limit is not None and (limit != 'None'):
        _args += ["--limit", f"{limit}"]
    print(f'running test_dist with args: {_args}')
    fpath, results = run_eval(*_args)
    print(results)

    if limit is None:
        # {
        # "results": {
        #     "hellaswag_d": {
        #     "acc": 0.30611431985660226,
        #     "acc_stderr": 0.004599358920909542,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_d": 0
        # },
        # "config": {
        #     "model": "dist_sim",
        #     "model_args": "WORD_AGG_SCHEME=relu|mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=23,PARALLELIZE=False",
        #     "task_args": "encoding_scheme=sentence_level_segmentation",
        #     "num_fewshot": 0,
        #     "batch_size": null,
        #     "device": "7",
        #     "no_cache": true,
        #     "limit": null,
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        # }
        expected = 0.3061
    elif limit == 100:
        expected = 0.24
    elif limit == 1000:
        expected = 0.287
    else:
        raise ValueError(f'Unsupported limit value {limit}')

    assert results['results']['hellaswag_d']['acc'] == expected, f"Test Failed: expected={expected}, got={results['results']['hellaswag_d']['acc']}"
    print('Test Passed')


if __name__ == '__main__':
    fire.Fire(_test)
