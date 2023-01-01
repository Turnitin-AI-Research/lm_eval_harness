"""Helper functions for notebooks"""
import os
import typing
import json
import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def remove_header(segment):
    if segment.startswith('model_args:'):
        return segment[len('model_args:'):]
    elif segment.startswith('task_args:'):
        return segment[len('task_args:'):]
    else:
        return segment


def parse_segment(segment) -> typing.List[str]:
    segment = remove_header(segment)
    kwargs = [kwarg for kwarg in segment.split(',') if kwarg]
    args = {}
    for kwarg_str in kwargs:
        k,v = kwarg_str.split('=')
        args[k] = v if v != 'None' else None
    return args


def parse_fname(fname: str) -> typing.Dict:
    args = {}
    for segment in fname.split('|'):
        args.update(parse_segment(segment))
    return args


def parse_file(fpath: str) -> typing.Dict:
    mtime = os.stat(fpath).st_mtime
    with open(fpath, 'rt') as f:
        o = json.load(f)
    task_version = o['versions']
    d = {'mtime': mtime}
    d['filename'] = Path(fpath).name
    for k, v in o['config'].items():
        if isinstance(v, str) and '=' in v:
            d.update(parse_segment(v))
        elif not v and k in ['model_args', 'task_args']:
            continue
        else:
            d[k] = v
    for task_name, results in o['results'].items():
        if task_name.endswith('_d'):
            task_name_out = task_name[:-len('_d')]
        elif task_name.endswith('_dg'):
            task_name_out = task_name[:-len('_dg')]
        else:
            task_name_out = task_name
        for k, v in results.items():
            # d[f'{task_name_out}_v{task_version[task_name]}:{k}'] = v
            d[f'{task_name_out}:{k}'] = v
    return d


def parse_dir(dirpath: str) -> pd.DataFrame:
    # pd.DataFrame([parse_fname(fname) for fname in os.listdir('lmeval_results')])
    fnames, mtimes = zip(*[(fentry.name, fentry.stat().st_mtime) for fentry in os.scandir(dirpath) if fentry.is_file() and fentry.name.endswith('.json')])
    return pd.DataFrame([parse_file(f'{dirpath}/{fname}') for fname in fnames])


def read_results(dir: str) -> pd.DataFrame:
    df = parse_dir(dir)
    df = df[[col for col in df.columns if col not in ['batch_size', 'device', 'no_cache', 'bootstrap_iters', 'description_dict']]]
    df = df[df.limit.isna()].assign(pretrained=df.pretrained.fillna('GPT2'))
    df = df.assign(model_type=df.model.map(lambda model: 'autoregressive' if model == 'gpt2' else (model))).drop(columns='model')
    return df


def task_metrics(df: pd.DataFrame, tasks: typing.List[str], *, sort_metrics=[], take_last=True) -> pd.DataFrame:
    metrics = tasks
    metrics_re = re.compile(r'^(' + r'|'.join([f'({m})' for m in metrics]) + ').*')
    print(f'metric cols regexp = {metrics_re}')
    model_cols = {'model_type', 'pretrained', 'ENCODING_LAYER', 'WORD_AGG_SCHEME', 'SEGMENT_AGG_SCHEME', 'EXAMPLE_AGG_SCHEME', 'NORM', 'SIMILARITY_FUNC', 'DECODING_SCHEME', 'STEER_VEC_INJ_LAYERS', 'STEER_VEC_INJ_POS'}
    model_cols = model_cols & set(df.columns)  # {col for col in model_cols if col in df.columns}
    task_cols = {'num_fewshot', 'encoding_scheme'} & set(df.columns)
    # metric_cols = {col for col in df.columns if metrics_re.fullmatch(col) is not None}
    task_metric_cols = {task: [col for col in df.columns if re.fullmatch(f'^{task}.*$', col)] for task in tasks}
    task_metric_cols = {task: cols for task, cols in task_metric_cols.items() if cols}  # remove empty lists
    metric_cols = {col for cols in task_metric_cols.values() for col in cols}
    assert metric_cols, f'No metrics found matching {metrics_re}'
    provenance_cols = {'mtime', 'filename'}
    selected_cols = task_cols | model_cols | metric_cols | provenance_cols
    assert selected_cols <= set(df.columns), f'Check that task_cols, model_cols, metri_cols and provenance_cols are all <= df.columns'
    if selected_cols < set(df.columns):
        print(f'Following columns will be dropped: {set(df.columns) - selected_cols}')
    if take_last:
        groupby_cols = (model_cols | task_cols)
        def _take_last(_df: pd.DataFrame) -> pd.DataFrame:
            _df = _df.sort_values(by='mtime', ascending=False)
            # return pd.Series({col: _df[col].dropna().iloc[0] if _df[col].dropna().shape[0] >=1 else None for col in _df.columns if col in metric_cols})
            metric_values = []
            for task in tasks:
                _task_metrics_df = _df[task_metric_cols[task]]
                _no_nans = _task_metrics_df.dropna()
                if len(_no_nans) > 0:
                    _task_metrics_sr = _no_nans.iloc[0]
                else:
                    # Couldn't find metrics
                    print(f'Did not find metrics: {_task_metrics_df.columns} for one group. Setting as NaN.')
                    _task_metrics_sr = _task_metrics_df.iloc[0]
                metric_values.append(_task_metrics_sr)
            # rows = [_df[task_metric_cols[task]].dropna().iloc[0] for task in tasks]
            return pd.concat(metric_values)
        df = df[list(selected_cols)].groupby(list(groupby_cols), dropna=False).agg(_take_last).dropna(how='all')
        df = df.reset_index(drop=False)
    else:
        df = df[list(selected_cols)].reset_index(drop=True)
        df = df.assign(date=pd.to_datetime(df.mtime, origin='unix', unit='s', utc=True).dt.tz_convert('US/Pacific'))
    if len(df) > 0 and sort_metrics:
        df = df.sort_values(by=sort_metrics, ascending=False)
    return df


def fig_parcats(df, main_metric, exclude_cols=[], *, height=700, width=None, remove_nonvariables=True, colorscale='cividis'):
    if remove_nonvariables:
        cols_to_remove = [col for col in df.columns if df[col].unique().shape[0] <= 1]
        cols_to_keep = [col for col in df.columns if df[col].unique().shape[0] > 1]
        if len(cols_to_remove) > 0:
            df_const = df[cols_to_remove].iloc[0].rename('Constant Variables')
            display(df_const)
        df = df[cols_to_keep]
    else:
        df = df
    fig = go.Figure(
        go.Parcats(arrangement='freeform', hoveron='color',
            dimensions=[{'label': col, 'values': df[col]} for col in df.columns if col not in exclude_cols + [main_metric]] + [{'label': main_metric, 'values': df[main_metric]}],
            line={
                'color': df[main_metric],
                'coloraxis': 'coloraxis'
                }
            )
    )
    # fig = px.parallel_categories(df, color='hellaswag:acc', dimensions=[col for col in df_hellaswag.columns if not col.startswith('hellaswag:acc')] + ['hellaswag:acc'],
    #                              color_continuous_scale=colorscale, height=700)
    layout_args = {
        'height': height,
        'coloraxis': {'colorscale': colorscale,
                      'showscale': True,
                      'colorbar': {'lenmode': 'fraction', 'len': 1.0, 'yanchor': 'top', 'y': 1.0}
                      },
        }
    if width is not None:
        layout_args['width'] = width
    fig.update_layout(layout_args)

    return fig
