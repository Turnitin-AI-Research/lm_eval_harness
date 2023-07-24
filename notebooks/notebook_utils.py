"""Helper functions for notebooks"""
import math
import os
import typing
import json
import re
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


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
        k, v = kwarg_str.split('=')
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
    fnames, mtimes = zip(*[(fentry.name, fentry.stat().st_mtime)
                         for fentry in os.scandir(dirpath) if fentry.is_file() and fentry.name.endswith('.json')])
    return pd.DataFrame([parse_file(f'{dirpath}/{fname}') for fname in fnames])


def read_results(dir: str) -> pd.DataFrame:
    df = parse_dir(dir)
    df = df[[col for col in df.columns if col not in ['batch_size',
                                                      'device', 'no_cache', 'bootstrap_iters', 'description_dict']]]
    df = df[df.limit.isna()].assign(pretrained=df.pretrained.fillna('GPT2'))
    df = df.assign(model_type=df.model.map(lambda model: 'autoregressive' if model
                   == 'gpt2' else (model))).drop(columns='model')
    df = fix_args(df)
    return df


def fix_args(df: pd.DataFrame) -> pd.DataFrame:
    def fix_row(row):
        if 'ENCODING_LAYER' in row and pd.isna(row.ENCODING_LAYER):
            row.ENCODING_LAYER = '-1'
        if 'OUT_WORD_AGG_SCHEME' in row and pd.isna(row.OUT_WORD_AGG_SCHEME):
            row.OUT_WORD_AGG_SCHEME = row.WORD_AGG_SCHEME
        if 'OUT_WORD_AGG_SCHEME' in row and pd.isna(row.OUT_ENCODING_LAYER):
            row.OUT_ENCODING_LAYER = row.ENCODING_LAYER
        return row
    return df.apply(fix_row, axis='columns')  # type: ignore


def drop_redundant_hellaswag_args(df: pd.DataFrame) -> pd.DataFrame:
    "Merge redundant runs of hellaswag"
    def fix_row(row):
        if row.encoding_scheme == 'concat_all_examples':
            row.SEGMENT_AGG_SCHEME = None
            row.EXAMPLE_AGG_SCHEME = None
        elif row.encoding_scheme == 'concat_each_example':
            row.SEGMENT_AGG_SCHEME = None
            if row.num_fewshot == 0:
                row.encoding_scheme = 'concat_all_examples'
                row.EXAMPLE_AGG_SCHEME = None
        elif row.encoding_scheme == 'sentence_level_segmentation':
            if row.num_fewshot == 0:
                row.EXAMPLE_AGG_SCHEME = None
        elif row.encoding_scheme == 'segment_each_example':
            if row.num_fewshot == 0:
                row.encoding_scheme = 'concat_all_examples'
                row.EXAMPLE_AGG_SCHEME = None
                row.SEGMENT_AGG_SCHEME = None
        return row
    df = df.apply(fix_row, axis='columns')  # type: ignore
    return df.drop_duplicates()


def task_metrics(df: pd.DataFrame, tasks: typing.List[str], *, sort_metrics=[], take_last=True) -> pd.DataFrame:
    metrics = tasks
    metrics_re = re.compile(r'^(' + r'|'.join([f'({m})' for m in metrics]) + ').*')
    print(f'metric cols regexp = {metrics_re}')
    model_cols = {'model_type', 'pretrained', 'ENCODING_LAYER', 'WORD_AGG_SCHEME', 'SEGMENT_AGG_SCHEME',
                  'EXAMPLE_AGG_SCHEME', 'NORM', 'SIMILARITY_FUNC', 'DECODING_SCHEME', 'STEER_VEC_INJ_LAYERS',
                  'STEER_VEC_INJ_POS', 'ADD_POS', 'OUT_ENCODING_LAYER', 'OUT_WORD_AGG_SCHEME'}
    model_cols = model_cols & set(df.columns)  # {col for col in model_cols if col in df.columns}
    task_cols = {'num_fewshot', 'encoding_scheme'} & set(df.columns)
    # metric_cols = {col for col in df.columns if metrics_re.fullmatch(col) is not None}
    task_metric_cols = {task: [col for col in df.columns if re.fullmatch(f'^{task}.*$', col)] for task in tasks}
    task_metric_cols = {task: cols for task, cols in task_metric_cols.items() if cols}  # remove empty lists
    metric_cols = {col for cols in task_metric_cols.values() for col in cols}
    assert metric_cols, f'No metrics found matching {metrics_re}'
    provenance_cols = {'mtime', 'filename'}
    selected_cols = task_cols | model_cols | metric_cols | provenance_cols
    assert selected_cols <= set(
        df.columns), f'Check that task_cols, model_cols, metri_cols and provenance_cols are all <= df.columns'
    if selected_cols < set(df.columns):
        print(f'Following columns will be dropped: {set(df.columns) - selected_cols}')
    if take_last:
        groupby_cols = (model_cols | task_cols)

        def _take_last(_df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(_df, pd.Series):
                raise ValueError
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
                    # print(f'Did not find metrics: {_task_metrics_df.columns} for one group. Setting as NaN.')
                    _task_metrics_sr = _task_metrics_df.iloc[0]
                metric_values.append(_task_metrics_sr)
            # rows = [_df[task_metric_cols[task]].dropna().iloc[0] for task in tasks]
            return pd.concat(metric_values)
        df = df[list(selected_cols)].groupby(list(groupby_cols), dropna=False,
                                             sort=False).apply(_take_last).dropna(how='all')
        df = df.reset_index(drop=False)
    else:
        df = df[list(selected_cols)].reset_index(drop=True)
        df = df.assign(date=pd.to_datetime(df.mtime, origin='unix', unit='s', utc=True).dt.tz_convert('US/Pacific'))
    if len(df) > 0 and sort_metrics:
        df = df.sort_values(by=sort_metrics, ascending=False)
    return df


def fig_parcats(df, main_metric, exclude_cols=[], *, height=700, width=None, remove_nonvariables=True, colorscale='cividis', title=None):
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
                   dimensions=[{'label': col, 'values': df[col]} for col in df.columns if col not in exclude_cols
                               + [main_metric]] + [{'label': main_metric, 'values': df[main_metric]}],
                   line={
                       'color': df[main_metric],
                       'coloraxis': 'coloraxis'
                   },
                   ),

    )
    layout_args = {
        'height': height,
        'coloraxis': {'colorscale': colorscale,
                      'showscale': True,
                      'colorbar': {'lenmode': 'fraction', 'len': 1.0, 'yanchor': 'top', 'y': 1.0}
                      },
        'title': title
    }
    if width is not None:
        layout_args['width'] = width
    fig.update_layout(layout_args)

    return fig


def compare_metrics(df1, df2):
    assert not set(df1.columns) - set(df2.columns)
    assert not set(df2.columns) - set(df1.columns)
    group_cols = list(set(df1.columns) - {'hellaswag:acc'})
    df1 = df1.groupby(group_cols, dropna=False, sort=False).max()
    df2 = df2.groupby(group_cols, dropna=False, sort=False).max()
    index = (df1.index & df2.index)
    print(f'overlapping count = {len(index)}')
    new_better, old_better, num_equal = 0, 0, 0
    old_better_by, new_better_by = [], []
    for id in index:
        id = pd.MultiIndex.from_tuples([id])
        val1 = df1.loc[id]['hellaswag:acc'].iloc[0]
        val2 = df2.loc[id]['hellaswag:acc'].iloc[0]
        if math.isclose(val1, val2):
            num_equal += 1
        elif (val1 > val2):
            old_better += 1
            old_better_by.append((val1 - val2) * 100)
        elif val2 > val1:
            new_better += 1
            new_better_by.append((val2 - val1) * 100)

    print(f'new_better = {(new_better/len(index))*100:.2f}%, old_better = {(old_better/len(index))*100:.2f}%, equal = {(num_equal/len(index))*100:.2f}%')
    print(f'new better by {np.mean(new_better_by):.3f}, old better by {np.mean(old_better_by):.3f} absolute % points')


def plot_hellaswag_pct2(df1, df2=None, num_fewshots=[0, 5], fig=None, col=None, row=None):
    if fig is None:
        fig = go.Figure(layout=dict(  # yaxis_range=[0, 1.1],
            yaxis_title='Normalized Hellaswag Score'))
    for num_few_shot in num_fewshots:
        for encoding_scheme in df1.encoding_scheme.unique():
            _df = df1[(df1.num_fewshot == num_few_shot) & (df1.encoding_scheme == encoding_scheme)]
            trace_name = f'K={num_few_shot}, {encoding_scheme if encoding_scheme != "sentence_level_segmentation" else "segment_sentences"}'
            if len(_df) > 0:
                fig.add_trace(go.Bar(x=_df.pretrained, y=_df['hellaswag:acc:pct'], name=trace_name), col=col, row=row)
            if df2 is not None:
                _df = df2[(df2.num_fewshot == num_few_shot) & (df2.encoding_scheme == encoding_scheme)]
                if len(_df) > 0:
                    fig.add_trace(go.Scatter(x=_df.pretrained, y=_df['hellaswag:acc:pct'],
                                             name=('(best ablation) ' + trace_name),
                                             mode='lines+markers',
                                             line=dict(dash='dot')),
                                  col=col, row=row)
    return fig


def plot_hellaswag_pct(df1, df2=None, num_fewshots=[0, 5], fig=None, col=None, row=None):
    if fig is None:
        fig = go.Figure(layout=dict(  # yaxis_range=[0, 1.1],
            yaxis_title='Normalized Hellaswag Score'))
    for num_few_shot in num_fewshots:
        for encoding_scheme in df1.encoding_scheme.unique():
            _df = df1[(df1.num_fewshot == num_few_shot) & (df1.encoding_scheme == encoding_scheme)]
            trace_name = f'K={num_few_shot}, {encoding_scheme if encoding_scheme != "sentence_level_segmentation" else "segment_sentences"}'
            if len(_df) > 0:
                fig.add_trace(go.Line(x=_df.pretrained, y=_df['hellaswag:acc:pct'],
                              name=trace_name, line=dict(width=1)), col=col, row=row)
            if df2 is not None:
                _df = df2[(df2.num_fewshot == num_few_shot) & (df2.encoding_scheme == encoding_scheme)]
                if len(_df) > 0:
                    fig.add_trace(go.Scatter(x=_df.pretrained, y=_df['hellaswag:acc:pct'],
                                             name=('(best ablation) ' + trace_name),
                                             mode='lines+markers',
                                             line=dict(dash='dot')),
                                  col=col, row=row)
    return fig


def subplot_hellaswag_pct(df, width=1000, height=300):
    fig = make_subplots(rows=2,
                        #   row_heights=ps.row_dist,
                        cols=1,
                        column_widths=[width],
                        row_heights=[height, height],
                        shared_yaxes='all',
                        shared_xaxes='all',  # if stack_heads else False,
                        #   horizontal_spacing=ps.horizontal_spacing,
                        #   vertical_spacing=ps.vertical_spacing,
                        #   column_titles=[head_name(i) for i in range(ps.num_heads)] if (
                        #       not stack_heads and ps.num_heads > 1) else [W_name] if _is_standalone else [f'${T_spec["latexName"]}$'],
                        row_titles=['K=0', 'K=5'],
                        y_title='Normalized Hellaswag Score',
                        #   specs=ps.subplot_specs,
                        #   x_title=xaxis_title
                        )

    fig.update_yaxes(range=[0, 1.], dtick=0.1, row=1, col=1)
    fig.update_yaxes(range=[0, 1.], dtick=0.1, row=2, col=1)
    # fig.update_layout(coloraxis=dict(colorscale='cividis_r'))
    plot_hellaswag_pct2(df, num_fewshots=[0], fig=fig, row=1, col=1)
    plot_hellaswag_pct2(df, num_fewshots=[5], fig=fig, row=2, col=1)
    return fig
