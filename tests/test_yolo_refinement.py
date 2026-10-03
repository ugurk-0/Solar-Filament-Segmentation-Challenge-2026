import numpy as np
import pytest
from yolo_refinement import refine_masks


def test_refinement_adds_only_nearby_supported_probability():
    mask = np.zeros((32, 32), bool)
    mask[10:20, 12:15] = True
    probability = np.zeros((32, 32), np.float32)
    probability[11:19, 10:17] = .8
    probability[0:3, 0:3] = 1
    valid = np.ones_like(mask)
    valid[:, 10] = False
    pred = refine_masks([mask], [.9], probability, valid, radius=3, min_area=1)[0]
    assert pred[14, 11] and not pred[14, 10]
    assert not pred[0:3, 0:3].any() and not pred[10, 12]
    assert (probability[pred] >= .5).all()


def test_refinement_preserves_disjoint_instance_ownership():
    mask = np.zeros((32, 32), bool)
    mask[10:20, 12:15] = True
    shifted = np.roll(mask, 4, axis=1)
    probability = np.ones(mask.shape, np.float32)
    pred = refine_masks([mask, shifted], [.9, .8], probability,
                        np.ones_like(mask), radius=3, min_area=1)
    assert len(pred) == 2 and not (pred[0] & pred[1]).any()
    assert refine_masks([mask], [.05], probability, np.ones_like(mask)) == []


def test_refinement_rejects_nonfinite_map():
    with pytest.raises(ValueError):
        refine_masks([], [], np.full((4, 4), np.nan), np.ones((4, 4), bool))
