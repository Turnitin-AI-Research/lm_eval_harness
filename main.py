import typing
import argparse
import os
import json
import logging
from pathlib import Path
import fnmatch
import hashlib
import torch

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--task_args", default="")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--limit", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args(args=None if not args else args)


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def results_fpath(*args) -> typing.Optional[str]:
    args = parse_args(*args)
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    if args.output_path:
        fpath = str(Path(args.output_path).resolve())
    elif args.output_dir:
        model_args = args.model_args.replace("/", ":")
        fname = (f"model={args.model}"
                 f"|tasks={','.join(task_names)}"
                 f"|model_args:{model_args}|task_args:{args.task_args}"
                 f"|num_fewshot={args.num_fewshot}|limit={args.limit}")
        fname = hashlib.shake_128(bytes(fname, encoding='utf-8')).hexdigest(20)
        fpath = f"{args.output_dir}/{fname}.json"
        fpath = str(Path(fpath).resolve())
    else:
        fpath = None
    return fpath


def main(*args):
    orig_args = args
    args = parse_args(*args)

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}, args: {args}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        task_args=args.task_args,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    fpath = results_fpath(*orig_args)
    if fpath is not None:
        with open(fpath, "wt", encoding='utf-8') as f:
            f.write(dumped)
            print(f'Saved output to {fpath}')
    else:
        print('Output will not be saved to file.')

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))
    return fpath, results


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    main()
