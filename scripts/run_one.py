import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from main import main

_args = [
    "--device", "7",
    "--output_dir", "lmeval_results_debug/",
    # "--limit", "1000",
    "--tasks", 'hellaswag_d',
    "--model", 'dist_sim',
    "--no_cache",
    '--num_fewshot', '5',
    '--task_args', 'encoding_scheme=concat_all_examples',  # merge_all_segments
    '--model_args', ('WORD_AGG_SCHEME=mean,EXAMPLE_AGG_SCHEME=None,SEGMENT_AGG_SCHEME=None,NORM=layer,SIMILARITY_FUNC=dot_product,pretrained=EleutherAI/gpt-neo-1.3B,'
                     + 'ENCODING_LAYER=middle'
                    #  + ',DECODING_SCHEME=steer_vec,STEER_VEC_INJ_LAYERS=10-12,STEER_VEC_INJ_POS=all'
                     )
]

fpath, results = main(*_args)
print(results)