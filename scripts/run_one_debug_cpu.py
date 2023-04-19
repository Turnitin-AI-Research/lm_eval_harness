import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from main import main

_args = [
    "--device", "cpu",
    "--output_dir", "lmeval_results_debug/",
    # "--limit", "1000",
    "--tasks", 'hellaswag_dg',
    "--model", 'dist_gen',
    "--no_cache",
    '--num_fewshot', '5',
    '--task_args', 'encoding_scheme=concat_all_examples',
    '--model_args', 'WORD_AGG_SCHEME=concat,ENCODING_LAYER=E,DECODING_SCHEME=parameterless_attention'
]

fpath, results = main(*_args)
print(results)
