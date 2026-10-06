from types import SimpleNamespace
import pytest
import torch
from yolo_model_identity import model_identity


def test_model_identity_checks_actual_architecture():
    network = torch.nn.Linear(2, 1)
    network.yaml = {'scale': 'n', 'yaml_file': 'yolo11n-seg.yaml'}
    model = SimpleNamespace(task='segment', model=network)
    assert model_identity(model, 'n')['parameters'] == 3
    with pytest.raises(ValueError, match='scale=n'):
        model_identity(model, 'm')
    model.task = 'detect'
    with pytest.raises(ValueError, match='task=detect'):
        model_identity(model, 'n')
