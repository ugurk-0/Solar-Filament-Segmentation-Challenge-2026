import json
import pytest
from yolo_recovery import record_protocol


def test_evaluation_recovery_preserves_training_record(tmp_path):
    path = tmp_path / 'protocol.json'
    old = dict(split=[1, 2], model='n', source_sha256={'solution.py': 'same', 'train_yolo_experiment.py': 'old'})
    record_protocol(path, old, {})
    original = path.read_bytes()
    new = {**old, 'source_sha256': {**old['source_sha256'], 'train_yolo_experiment.py': 'new'}}
    record_protocol(path, new, {'training_complete': True}, evaluation_only=True)
    assert path.read_bytes() == original
    assert json.loads((tmp_path / 'evaluation_recovery_protocol.json').read_text())['evaluation_protocol'] == new


def test_recovery_refuses_changed_split_or_metric(tmp_path):
    path = tmp_path / 'protocol.json'
    old = dict(split=[1], source_sha256={'solution.py': 'original'})
    record_protocol(path, old, {})
    with pytest.raises(ValueError, match='split'):
        record_protocol(path, {**old, 'split': [2]}, {'training_complete': True}, True)
    with pytest.raises(ValueError, match='metric'):
        record_protocol(path, {**old, 'source_sha256': {'solution.py': 'changed'}}, {'training_complete': True}, True)
    with pytest.raises(ValueError, match='completed training'):
        record_protocol(path, old, {}, True)
