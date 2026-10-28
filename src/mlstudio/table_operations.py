"""Bounded, explicit table combination and aggregation for visual workflows."""
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

MAX_ROWS = 20000
MAX_COLUMNS = 300


def _check(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError('Use a nonempty table with distinct column names.')
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError('Table operations support up to 20,000 rows and 300 columns.')


def _bounded(frame):
    _check(frame)
    return frame.reset_index(drop=True)


def combine_tables(left, right, kind, params):
    _check(left); _check(right)
    if kind == 'Concatenate':
        if len(left) + len(right) > MAX_ROWS:
            raise ValueError('Combined tables exceed 20,000 rows. Sample inputs first.')
        mode = params.get('columns', 'union')
        if mode not in {'union', 'intersection'}:
            raise ValueError('Choose all columns or shared columns for concatenation.')
        shared = [c for c in left if c in right]
        if mode == 'intersection' and not shared:
            raise ValueError('These tables have no shared column names.')
        for c in shared:
            a = is_numeric_dtype(left[c]) and not is_bool_dtype(left[c])
            b = is_numeric_dtype(right[c]) and not is_bool_dtype(right[c])
            if a != b and left[c].notna().any() and right[c].notna().any():
                raise ValueError(f'Column {c!r} has different numeric/category types. Correct its type before combining.')
        a, b = left.copy(), right.copy()
        source = params.get('source_column', '')
        if not isinstance(source, str) or len(source) > 100:
            raise ValueError('Use a source column name up to 100 characters.')
        if source:
            if source in left or source in right:
                raise ValueError('The source column name already exists. Choose another name.')
            a[source], b[source] = 'Input 1', 'Input 2'
        return _bounded(pd.concat([a, b], join='outer' if mode == 'union' else 'inner', ignore_index=True, sort=False))
    if kind != 'Merge data':
        raise ValueError('Choose Merge data or Concatenate.')
    lk, rk = params.get('left_key'), params.get('right_key')
    how = params.get('how', 'left')
    if lk not in left or rk not in right or how not in {'left', 'inner', 'outer'}:
        raise ValueError('Choose a matching column from each input and a supported join type.')
    if left[lk].isna().any() or right[rk].isna().any():
        raise ValueError('Matching keys contain missing values. Review those rows before merging.')
    if right[rk].duplicated().any():
        raise ValueError('Input 2 has repeated matching keys. Make its keys unique to avoid multiplying rows.')
    try:
        output = left.merge(right, how=how, left_on=lk, right_on=rk, suffixes=('', '_extra'), validate='many_to_one', sort=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('Matching columns need compatible types and unambiguous output names. '+str(exc)) from exc
    return _bounded(output)


def summarize_table(frame, kind, params):
    _check(frame)
    group, value = params.get('group'), params.get('value')
    statistic = params.get('statistic', 'count')
    if group not in frame or statistic not in {'count', 'mean', 'median', 'sum', 'min', 'max', 'std'}:
        raise ValueError('Choose a grouping column and a supported statistic.')
    if frame[group].nunique(dropna=False) > 200:
        raise ValueError('Use a grouping column with at most 200 groups.')
    if statistic != 'count' and (value not in frame or not is_numeric_dtype(frame[value]) or is_bool_dtype(frame[value])):
        raise ValueError('Choose a numeric measurement for this statistic.')
    if kind == 'Group by':
        groups = frame.groupby(group, dropna=False, observed=True, sort=False)
        result = groups.size() if statistic == 'count' else groups[value].agg(statistic)
        name = 'Row count' if statistic == 'count' else f'{statistic}({value})'
        if name == group: name = 'Summary ' + name
        return _bounded(result.rename(name).reset_index())
    if kind != 'Pivot table': raise ValueError('Choose Group by or Pivot table.')
    column = params.get('column')
    if column not in frame or column == group:
        raise ValueError('Choose a different column for the pivot headings.')
    if frame[column].nunique(dropna=False) > 100:
        raise ValueError('Use a pivot heading column with at most 100 values.')
    groups = frame.groupby([group, column], dropna=False, observed=True, sort=False)
    result = groups.size() if statistic == 'count' else groups[value].agg(statistic)
    output = result.unstack(fill_value=0 if statistic == 'count' else None)
    output.columns = [f'{column}={v}' for v in output.columns]
    if not output.columns.is_unique: raise ValueError('Pivot heading values have ambiguous text labels. Convert the heading column to one consistent type.')
    return _bounded(output.reset_index())
