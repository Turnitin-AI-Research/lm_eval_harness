from typing import Optional, Union
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import fire
from main import main as run_eval


def _test(limit: int = 100, device: str = '6', model_parallel: bool = False):
    if model_parallel:
        device = 'cpu'
        print('Setting device to "cpu" for model-parallel mode')
    _args = [
        "--device", str(device),
        "--output_dir", "lmeval_results_debug/",
        "--tasks", 'hellaswag_d',
        "--model", 'dist_sim',
        "--no_cache",
        '--num_fewshot', '5',
        '--task_args', 'encoding_scheme=concat_all_examples',  # merge_all_segments
        '--model_args', ('WORD_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B'
                         + ',ENCODING_LAYER=middle'
                         + f',PARALLELIZE={model_parallel}'
                         #  + ',DECODING_SCHEME=steer_vec,STEER_VEC_INJ_LAYERS=10-12,STEER_VEC_INJ_POS=all'
                         )
    ]
    if limit is not None:
        _args += ["--limit", f"{limit}"]
    print(f'running test_dist with args: {_args}')
    fpath, results = run_eval(*_args)
    print(results)
    if limit == 1000:
        # "hellaswag_d": {
        #   "acc": 0.297,
        #   "acc_stderr": 0.014456832294801105,
        #   "rand_acc": 0.25,
        #   "rand_acc_stderr": 0.0
        # }
        # "results": {
        #     "hellaswag_d": {
        #     "acc": 0.298,
        #     "acc_stderr": 0.014470846741134712,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_d": 0
        # },
        # "config": {
        #     "model": "dist_sim",
        #     "model_args": "WORD_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=middle",
        #     "task_args": "encoding_scheme=concat_all_examples",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "7",
        #     "no_cache": true,
        #     "limit": "1000",
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        expected = 0.298
    elif limit == 100:
        # "hellaswag_d": {
        #   "acc": 0.33,
        #   "acc_stderr": 0.04725815626252605,
        #   "rand_acc": 0.25,
        #   "rand_acc_stderr": 0.0
        # }
        # "results": {
        #     "hellaswag_d": {
        #     "acc": 0.34,
        #     "acc_stderr": 0.04760952285695235,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_d": 0
        # },
        # "config": {
        #     "model": "dist_sim",
        #     "model_args": "WORD_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=middle",
        #     "task_args": "encoding_scheme=concat_all_examples",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "7",
        #     "no_cache": true,
        #     "limit": "100",
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        expected = 0.34
    elif limit == 10:
        # "hellaswag_d": {
        #   "acc": 0.2,
        #   "acc_stderr": 0.13333333333333333,
        #   "rand_acc": 0.25,
        #   "rand_acc_stderr": 0.0
        # }
        # "results": {
        #     "hellaswag_d": {
        #     "acc": 0.2,
        #     "acc_stderr": 0.13333333333333333,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_d": 0
        # },
        # "config": {
        #     "model": "dist_sim",
        #     "model_args": "WORD_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=middle",
        #     "task_args": "encoding_scheme=concat_all_examples",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "7",
        #     "no_cache": true,
        #     "limit": "10",
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        expected = 0.2
    elif limit is None:
        # "hellaswag_d": {
        #     "acc": 0.29117705636327423,
        #     "acc_stderr": 0.004533764686211992,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0
        #     }
        expected = 0.291177

    assert results['results']['hellaswag_d']['acc'] == expected, f"Test Failed: expected={expected}, got={results['results']['hellaswag_d']['acc']}"
    print('Test Passed')

if __name__ == '__main__':
    fire.Fire(_test)
