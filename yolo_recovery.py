"""Preserve training provenance when explicitly recovering completed evaluation."""
import hashlib
import json
from pathlib import Path
import solution as sol


def record_protocol(path, protocol, state, evaluation_only=False):
    path = Path(path)
    if evaluation_only and not state.get('training_complete'):
        raise ValueError('Evaluation-only recovery requires completed training')
    if not path.exists():
        if evaluation_only:
            raise ValueError('Missing original training protocol')
        sol._write_json_atomically(path, protocol)
        return
    old = json.loads(path.read_text())
    if old == protocol and not evaluation_only:
        return
    if not evaluation_only:
        raise ValueError('Inputs changed; choose a new --output directory')
    old_inputs = {key: value for key, value in old.items() if key != 'source_sha256'}
    new_inputs = {key: value for key, value in protocol.items() if key != 'source_sha256'}
    if old_inputs != new_inputs:
        raise ValueError('Recovery cannot change model, split, baseline or training settings')
    allowed = {'train_yolo_experiment.py', 'experiment_yolo.py', 'scripts/update_yolo_readme.py', 'yolo_recovery.py'}
    changed_core = [key for key, value in old.get('source_sha256', {}).items()
                    if key not in allowed and protocol.get('source_sha256', {}).get(key) != value]
    if changed_core:
        raise ValueError(f'Recovery requires unchanged data/model/metric sources: {changed_core}')
    recovery = dict(original_protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    evaluation_protocol=protocol)
    recovery_path = path.with_name('evaluation_recovery_protocol.json')
    if recovery_path.exists() and json.loads(recovery_path.read_text()) != recovery:
        raise ValueError('An evaluation recovery with different inputs already exists')
    sol._write_json_atomically(recovery_path, recovery)
