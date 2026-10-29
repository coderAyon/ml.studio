"""Real SQLite operations must commit/rollback and release every connection."""
import gc
import json
import warnings

import pytest
from mlstudio.storage import Store


def test_history_connections_close_on_reads_writes_and_failed_serialization(tmp_path):
    gc.collect()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ResourceWarning)
        store = Store(tmp_path)
        store.add('Decision Tree', {'accuracy': .75})
        store.record_final_exposure('dataset-a', 'split-a')
        assert store.final_exposure_count('dataset-a') == 1
        assert store.final_exposure_count('dataset-b') == 0
        assert json.loads(store.history()[0][2]) == {'accuracy': .75}
        with pytest.raises(ValueError): store.add('invalid', {'score': float('nan')})
        assert len(Store(tmp_path).history()) == 1
        gc.collect()
    assert not [warning for warning in captured if 'unclosed database' in str(warning.message)]
