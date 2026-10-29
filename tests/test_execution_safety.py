"""Preflight must reflect real dense preparation, and unsupported nodes fail closed."""
import numpy as np
import pandas as pd
import pytest

from mlstudio.research import ResearchConfig, ResearchSession
from mlstudio.workflow import Graph


def test_research_rejects_actual_encoder_width_before_fitting():
    frame = pd.DataFrame({f'category{i}': [f'v{j % 80}' for j in range(15000)] for i in range(5)})
    frame['label'] = [0, 1] * 7500
    with pytest.raises(ValueError, match='memory budget'):
        ResearchSession(frame, ResearchConfig(target='label'))


def test_research_accounts_for_numeric_missing_indicator_columns():
    frame = pd.DataFrame(np.ones((16000, 180)), columns=[f'x{i}' for i in range(180)])
    frame.iloc[0, :] = np.nan; frame['label'] = [0, 1] * 8000
    with pytest.raises(ValueError, match='memory budget'):
        ResearchSession(frame, ResearchConfig(target='label', preparation={'missing_indicators': True}))


def test_research_records_effective_seed_and_preparation():
    frame = pd.DataFrame({'x': np.arange(80, dtype=float), 'label': [0, 1] * 40})
    session = ResearchSession(frame, ResearchConfig(target='label', seed=17, repeats=1,
        preparation={'scaling': 'minmax', 'imputation': 'mean'}))
    result = session.compare(['Decision Tree'])['models'][0]
    assert result['parameters']['random_state'] == 17
    assert result['preparation']['scaling'] == 'minmax'
    assert result['preparation']['imputation'] == 'mean'


@pytest.mark.parametrize('dtype', ['bool', 'boolean'])
def test_research_boolean_features_use_the_same_preparation_as_guided_training(dtype):
    values = [True, False] * 40
    if dtype == 'boolean': values[0] = None
    frame = pd.DataFrame({'flag': pd.Series(values, dtype=dtype), 'label': [0, 1] * 40})
    original = frame.copy(deep=True)
    session = ResearchSession(frame, ResearchConfig(target='label', repeats=1))
    report = session.compare(['Decision Tree'])
    assert len(report['models'][0]['scores']) == 3
    assert all(0 <= score <= 1 for score in report['models'][0]['scores'])
    assert session.final_used is False
    pd.testing.assert_frame_equal(frame, original)


def test_legacy_table_node_never_silently_removes_rows():
    frame = pd.DataFrame({'x': [1, 1, 2]})
    graph = Graph({'1': {'type': 'Dataset'}, '2': {'type': 'Data table'}}, [['1', '2']])
    pd.testing.assert_frame_equal(graph.execute(frame, '', 'Decision Tree')['2'], frame)


@pytest.mark.parametrize('kind', ['Normalize', 'Impute missing', 'Histogram', 'Select features'])
def test_legacy_executor_rejects_unsupported_steps_instead_of_deduplicating(kind):
    frame = pd.DataFrame({'x': [1, 1, 2]})
    graph = Graph({'1': {'type': 'Dataset'}, '2': {'type': kind}}, [['1', '2']])
    with pytest.raises(ValueError, match='reactive workflow'):
        graph.execute(frame, '', 'Decision Tree')
