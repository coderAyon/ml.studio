"""Results belong to a retained revision, never a reused history position."""
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QObject

from mlstudio.association import mine_rules
from mlstudio.comparison_tools import ComparisonTools
from mlstudio.data import DatasetVersions, Transformation
from mlstudio.flow_controller import FlowCoordinator
from mlstudio.workflow import Graph


def dataset():
    return DatasetVersions(pd.DataFrame({
        'x': range(80), 'category': ['A', 'B'] * 40,
        'colour': ['red', 'blue'] * 40, 'label': [0, 1] * 40,
    }))


def test_replacement_branch_has_new_revision_and_undo_redo_restores_identity():
    versions = dataset()
    original = versions.current
    first = versions.revision_id
    versions.apply(Transformation('Filter rows', ['category'], 'A'))
    branch_a = versions.revision_id
    assert first != branch_a
    versions.undo()
    assert versions.revision_id == first
    versions.redo()
    assert versions.revision_id == branch_a
    versions.undo()
    versions.apply(Transformation('Filter rows', ['category'], 'B'))
    assert versions.index == 1 and versions.revision_id not in {first, branch_a}
    branch_b = versions.revision_id
    versions.redo()
    assert versions.revision_id == branch_b
    with pytest.raises(ValueError, match='leave no data'):
        versions.apply(Transformation('Filter rows', ['category'], 'absent'))
    assert versions.index == 1 and versions.revision_id == branch_b and len(versions.versions) == 2
    with pytest.raises(AttributeError):
        versions.revision_id = first
    pd.testing.assert_frame_equal(versions.versions[0], original)


class Bridge(QObject):
    def __init__(self, versions):
        super().__init__()
        nodes = {'1': {'type': 'Dataset', 'position': [0, 0], 'params': {'source': 'workspace'}},
                 '2': {'type': 'Data table', 'position': [200, 0]},
                 '3': {'type': 'Decision Tree', 'position': [200, 200]}}
        graph = Graph(nodes, [['1', '2'], ['1', '3']])
        self.controller = SimpleNamespace(
            dataset=versions, canvas=SimpleNamespace(snapshot=lambda: graph),
            target=SimpleNamespace(currentText=lambda: 'label'),
            algorithm=SimpleNamespace(currentText=lambda: 'Decision Tree'), result=None,
            store=SimpleNamespace(add=lambda *args: None), refresh_history=lambda: None,
            require_data=lambda: versions.current)
        self.research = SimpleNamespace(_background=self.background)

    def training_preparation(self):
        return None

    def on_status(self, message):
        pass

    def announce(self, message):
        pass

    def background(self, work, message, complete):
        complete(work())


def test_replacement_branch_invalidates_workflow_outputs_export_and_model_activation():
    versions = dataset()
    # Both branches retain both classes so a real model can be fitted.
    versions.apply(Transformation('Select row positions', [], str(list(range(40)))))
    bridge = Bridge(versions)
    flow = FlowCoordinator(bridge)
    flow.run()
    assert flow.public['nodes']['3']['status'] == 'ready'
    first_signature = flow.last_signature
    assert flow.engine.values['2'].frame.x.iloc[0] == 0
    versions.undo()
    flow.state()  # Leaving Workflow does not discard retained engine values.
    versions.apply(Transformation('Select row positions', [], str(list(range(40, 80)))))
    assert flow.signature() != first_signature and flow.state()['stale']
    with pytest.raises(ValueError, match='Run the current workflow'):
        flow.command('flow_export', {'id': '2'})
    with pytest.raises(ValueError, match='Workflow changed'):
        flow.command('flow_activate', {'id': '3'})
    flow.run()
    assert flow.engine.values['2'].frame.x.iloc[0] == 40
    assert flow.public['computed'] == 3
    flow.command('flow_activate', {'id': '3'})
    assert bridge.controller.result.metadata['training_rows'] == 30
    flow.run(force=False)
    assert not flow.state()['stale']


def test_association_rules_cannot_revive_on_replacement_branch():
    versions = dataset()
    versions.apply(Transformation('Filter rows', ['category'], 'A'))
    tools = ComparisonTools(Bridge(versions))
    tools.rules = mine_rules(versions.current, {'columns': ['category', 'colour']})
    tools.rules_key = tools.key()
    assert tools.rules['rules']
    versions.undo()
    assert tools.state()['association']['result'] is None
    versions.apply(Transformation('Filter rows', ['category'], 'B'))
    assert tools.state()['association']['result'] is None
    with pytest.raises(ValueError, match='current association rules'):
        tools.command('association_export', {})


def test_async_rules_completion_rejects_replacement_branch():
    versions = dataset()
    versions.apply(Transformation('Filter rows', ['category'], 'A'))
    bridge = Bridge(versions)
    pending = []
    bridge.research._background = lambda work, message, complete: pending.append((work, complete))
    tools = ComparisonTools(bridge)
    tools.command('association_run', {'columns': ['category', 'colour']})
    report = pending[0][0]()
    versions.undo()
    versions.apply(Transformation('Filter rows', ['category'], 'B'))
    pending[0][1](report)
    assert tools.state()['association']['result'] is None and tools.rules is None


def test_async_workflow_completion_does_not_publish_or_record_an_obsolete_revision():
    versions = dataset()
    bridge = Bridge(versions)
    pending, history = [], []
    bridge.research._background = lambda work, message, complete: pending.append((work, complete))
    bridge.controller.store.add = lambda *args: history.append(args)
    flow = FlowCoordinator(bridge)
    flow.run()
    result = pending[0][0]()
    versions.apply(Transformation('Filter rows', ['category'], 'B'))
    pending[0][1](result)
    assert flow.last_signature is None and flow.state()['outputs'] == {} and history == []
