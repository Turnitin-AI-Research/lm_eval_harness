from typing import Optional, Union
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import fire
from main import main as run_eval


def _test(limit: int = 100, device: str = "7", model_parallel: bool = False):
    if model_parallel:
        device = 'cpu'
        print('Setting device to "cpu" for model-parallel mode')
    _args = [
        "--device", str(device),
        "--output_dir", "lmeval_results_debug/",
        "--tasks", 'hellaswag_dg',
        "--model", 'dist_gen',
        "--no_cache",
        '--num_fewshot', '5',
        '--task_args', 'encoding_scheme=sentence_level_segmentation',  # merge_all_segments
        '--model_args', ('WORD_AGG_SCHEME=-relu|mean,EXAMPLE_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=None,pretrained=EleutherAI/gpt-neo-1.3B'
                         + ',ENCODING_LAYER=0'
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
        #     "hellaswag_dg": {
        #     "acc": 0.31826329416450905,
        #     "acc_stderr": 0.004648503177353929,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0,
        #     "acc_norm": 0.3552081258713404,
        #     "acc_norm_stderr": 0.00477598265035592,
        #     "em": 0.0,
        #     "em_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_dg": 0
        # },
        # "config": {
        #     "model": "dist_gen",
        #     "model_args": "WORD_AGG_SCHEME=-relu|mean,EXAMPLE_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=None,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=0",
        #     "task_args": "encoding_scheme=sentence_level_segmentation",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "7",
        #     "no_cache": true,
        #     "limit": null,
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        # }
        expected = 0.3183
    elif limit == 1000:
        # {
        # "results": {
        #     "hellaswag_dg": {
        #     "acc": 0.337,
        #     "acc_stderr": 0.014955087918653596,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0,
        #     "acc_norm": 0.365,
        #     "acc_norm_stderr": 0.0152317762262649,
        #     "em": 0.0,
        #     "em_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_dg": 0
        # },
        # "config": {
        #     "model": "dist_gen",
        #     "model_args": "WORD_AGG_SCHEME=-relu|mean,EXAMPLE_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=None,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=0",
        #     "task_args": "encoding_scheme=sentence_level_segmentation",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "6",
        #     "no_cache": true,
        #     "limit": "1000",
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        # }
        expected = 0.337
    elif limit == 100:
        # {
        # "results": {
        #     "hellaswag_dg": {
        #     "acc": 0.34,
        #     "acc_stderr": 0.047609522856952344,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0,
        #     "acc_norm": 0.34,
        #     "acc_norm_stderr": 0.04760952285695236,
        #     "em": 0.0,
        #     "em_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_dg": 0
        # },
        # "config": {
        #     "model": "dist_gen",
        #     "model_args": "WORD_AGG_SCHEME=-relu|mean,EXAMPLE_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=None,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=0",
        #     "task_args": "encoding_scheme=sentence_level_segmentation",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "6",
        #     "no_cache": true,
        #     "limit": "100",
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        # }
        expected = 0.34
    elif limit == 10:
        # {
        # "results": {
        #     "hellaswag_dg": {
        #     "acc": 0.3,
        #     "acc_stderr": 0.15275252316519466,
        #     "rand_acc": 0.25,
        #     "rand_acc_stderr": 0.0,
        #     "acc_norm": 0.5,
        #     "acc_norm_stderr": 0.16666666666666666,
        #     "em": 0.0,
        #     "em_stderr": 0.0
        #     }
        # },
        # "versions": {
        #     "hellaswag_dg": 0
        # },
        # "config": {
        #     "model": "dist_gen",
        #     "model_args": "WORD_AGG_SCHEME=-relu|mean,EXAMPLE_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=None,pretrained=EleutherAI/gpt-neo-1.3B,ENCODING_LAYER=0",
        #     "task_args": "encoding_scheme=sentence_level_segmentation",
        #     "num_fewshot": 5,
        #     "batch_size": null,
        #     "device": "7",
        #     "no_cache": true,
        #     "limit": "10",
        #     "bootstrap_iters": 100000,
        #     "description_dict": {}
        # }
        # }
        expected = 0.3

    assert results['results']['hellaswag_dg']['acc'] == expected, f"Test Failed: expected={expected}, got={results['results']['hellaswag_dg']['acc']}"
    print('Test Passed')


if __name__ == '__main__':
    fire.Fire(_test)
