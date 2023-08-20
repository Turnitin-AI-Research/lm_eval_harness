import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from main import main

_args = [
    "--device", "1",
    "--output_dir", "lmeval_results_debug/",
    # "--limit", "1000",
    "--tasks", 'hellaswag_dg',
    "--model", 'dist_gen',
    "--no_cache",
    '--num_fewshot', '5',
    '--task_args', 'encoding_scheme=cross_encoding',
    '--model_args', 'pretrained=EleutherAI/gpt-neo-1.3B'
]

fpath, results = main(*_args)
print(results)