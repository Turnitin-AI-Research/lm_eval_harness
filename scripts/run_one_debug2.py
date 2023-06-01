import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

from main import main

_args = [
    "--device", "3",
    "--output_dir", "lmeval_results_debug/",
    # "--limit", "1000",
    "--tasks", 'hellaswag_dg',
    "--model", 'dist_gen',
    "--no_cache",
    '--num_fewshot', '5',
    # '--task_args', 'encoding_scheme=sentence_level_segmentation',
    # '--model_args', 'pretrained=EleutherAI/gpt-neo-1.3B,WORD_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=mean'
    '--task_args', 'encoding_scheme=sentence_level_segmentation',
    '--model_args', 'pretrained=EleutherAI/gpt-neo-2.7B,WORD_AGG_SCHEME=mean,SEGMENT_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=mean,ENCODING_LAYER=E'
]

fpath, results = main(*_args)
print(results)
