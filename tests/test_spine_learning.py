import numpy as np
import pytest
import torch
from spine_learning import spine_heatmap, auxiliary_loss, ObservationDataset


def test_spine_xy_alignment_foreground_and_support():
    foreground = np.zeros((40, 60), bool)
    foreground[7:14, 20:46] = True
    valid = np.ones_like(foreground)
    valid[:, 40:] = False
    target = spine_heatmap([{'spine': [21, 10, 44, 10]}], foreground, valid)
    assert target[10, 30] == 1 and target[11, 30] < 1
    assert not target[~foreground].any() and not target[~valid].any()
    assert target.dtype == np.float32 and np.isfinite(target).all()
    assert target.max() <= 1


def test_empty_targets_and_bad_annotation():
    foreground = np.ones((20, 20), bool)
    assert not spine_heatmap([], foreground, foreground).any()
    with pytest.raises(ValueError):
        spine_heatmap([{'spine': [1, 2, 3]}], foreground, foreground)


def test_aux_loss_finite_gradients_and_ignores_unsupported_pixels():
    logits = torch.zeros((1, 1, 10, 10), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 4, 4] = 1
    valid = torch.ones_like(logits)
    valid[:, :, :2] = 0
    a = auxiliary_loss(logits, target, valid)
    changed = target.clone()
    changed[:, :, :2] = 1
    assert torch.equal(a, auxiliary_loss(logits, changed, valid))
    a.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 0, 4, 4] < 0
    assert not logits.grad[:, :, :2].any()


def test_observation_frequency_and_actual_annotation_selection(monkeypatch):
    from types import SimpleNamespace
    import solution as sol
    infos = {'a': {'file_name':'one-Sun1.jpeg'}, 'b': {'file_name':'two-Sun1.jpeg'},
             'c': {'file_name':'one-Sun2.jpeg'}}
    coco = SimpleNamespace(imgs=infos)
    cfg = SimpleNamespace(aux_spine_weight=0)
    dataset = ObservationDataset(coco, list(infos), cfg, True, np.ones((8, 8), bool))
    assert len(dataset) == 2
    visited = []
    def raw_load(self, image_id):
        visited.append(image_id)
        return np.zeros((8, 8)), np.zeros((8, 8), bool), np.zeros((8, 8))
    monkeypatch.setattr(sol.FilamentDataset, '_load_full', raw_load)
    monkeypatch.setattr('spine_learning.random.choice', lambda members: members[-1])
    dataset._load_full('a')
    assert visited == ['b']
