import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

from main import main

_args = [
    "--device", "1",
    "--output_dir", "lmeval_results_debug/",
    # "--limit", "1000",
    "--tasks", 'hellaswag_dg',
    "--model", 'dist_gen',
    "--no_cache",
    '--num_fewshot', '5',
    '--task_args', 'encoding_scheme=concat_all_examples',
    '--model_args', 'pretrained=EleutherAI/gpt-neox-20B,WORD_AGG_SCHEME=concat,ENCODING_LAYER=E,DECODING_SCHEME=parameterless_attention'
]

fpath, results = main(*_args)
print(results)
