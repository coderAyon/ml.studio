"""Cleaning must preview real changed cells, including nominal category domains."""
import json
import pandas as pd
import pytest

from mlstudio.analyzer import analyze
from mlstudio.data import DatasetVersions, Transformation
from mlstudio.preprocessing import inspect_finding, preview_change


@pytest.mark.parametrize('operation,values,categories,expected,changed', [
    ('Trim whitespace', [' A', 'b', None], [' A', 'b', 'unused'], ['A', 'b', None], 1),
    ('Lowercase labels', ['A', 'b', None], ['A', 'b', 'unused'], ['a', 'b', None], 1),
    ('Fill missing', [None, None, None], [], ['Missing'] * 3, 3),
])
def test_nominal_cleaning_preview_apply_undo_redo(operation, values, categories, expected, changed):
    frame = pd.DataFrame({'label': pd.Categorical(values, categories=categories),
                          'untouched': pd.Series([1, None, 3], dtype='Int64')}, index=[0, 1, 2])
    original = frame.copy(deep=True)
    preview = preview_change(frame, operation, 'label')
    assert preview['summary']['changed_cells'] == changed
    assert preview['summary']['affected_rows'] == changed
    json.dumps(preview, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)
    versions = DatasetVersions(frame)
    versions.apply(Transformation(operation, ['label']))
    assert versions.current.label.astype(object).where(versions.current.label.notna(), None).tolist() == expected
    pd.testing.assert_series_equal(versions.current.untouched, original.untouched)
    applied = versions.current.copy(deep=True)
    versions.undo(); pd.testing.assert_frame_equal(versions.current, original)
    versions.redo(); pd.testing.assert_frame_equal(versions.current, applied)


@pytest.mark.parametrize('dtype', ['object', 'string', 'category'])
@pytest.mark.parametrize('title,values,positions', [
    ('Possibly mixed numeric values', ['1', '2', '3', '4', 'bad', None], [5]),
    ('Equivalent category labels', ['A', 'a', 'b', 'c', None], [1, 2]),
    ('Surrounding whitespace', [' a', 'b', None], [1]),
])
def test_badge_matches_exact_evidence_with_missing_and_duplicate_indices(dtype, title, values, positions):
    frame = pd.DataFrame({'label': pd.Series(values, dtype=dtype)})
    frame.index = ['same'] * len(frame)
    original = frame.copy(deep=True)
    issue = next(i for i in analyze(frame) if i.title == title)
    detail = inspect_finding(frame, issue)
    assert issue.count == detail['affected_count'] == len(positions)
    assert detail['preview']['positions'] == positions
    pd.testing.assert_frame_equal(frame, original)
