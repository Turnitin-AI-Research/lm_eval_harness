import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import itertools
from main import main


results_dir = "lmeval_results_t5/"
num_fewshots = [5]
task_models = [('hellaswag_dg', 'dist_gen'), ('webqs_dg', 'dist_gen')]  # ('hellaswag_d', 'dist_sim'), ('webqs_dg', 'dist_gen')]
pretrained = ['google/flan-t5-xl']
# ['merge_all_segments', 'segment_each_example', 'concat_each_example', 'concat_all_examples']
encoding_schemes = ['sentence_level_segmentation', 'segment_each_example', 'concat_each_example']
# ['-relu|mean', '-relu+|mean', 'relu+|mean', 'relu|mean', 'relu+|last', 'relu|last', '-relu+|last', 'relu+|last']
word_agg_schemes = ['-relu|mean', '-relu+|mean', 'mean', 'relu+|mean', 'relu|mean']
segment_agg_schemes = [None, 'mean']
example_agg_schemes = [None, 'mean']
norms = ['layer', None]
sim_funcs = ['dot_product']
# ['middle', None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
encoding_layers = [0]
parallelize = True

if 0 in num_fewshots:
    ALLOWED_ZEROSHOT_ENCODING_SCHEMES = {'segment_each_example', 'sentence_level_segmentation'}
    ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES = {None}
    assert ALLOWED_ZEROSHOT_EXAMPLE_AGG_SCHEMES & set(example_agg_schemes)
    assert ALLOWED_ZEROSHOT_ENCODING_SCHEMES & set(encoding_schemes)

responses = []
os.makedirs(results_dir, exist_ok=True)
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
    model_args = (f'WORD_AGG_SCHEME={word_agg_scheme},EXAMPLE_AGG_SCHEME={example_agg_scheme},SEGMENT_AGG_SCHEME={segment_agg_scheme}' +
                  f',NORM={norm},SIMILARITY_FUNC={sim_func},ENCODING_LAYER={encoding_layer},PARALLELIZE={parallelize}')
    if submodel is not None:
        model_args = model_args + f',pretrained={submodel}'
    _args.extend(['--model_args', model_args])
    response = main(*_args)
    responses.append(response[1])

print(responses)
