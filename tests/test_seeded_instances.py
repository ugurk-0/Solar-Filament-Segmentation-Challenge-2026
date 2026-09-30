import numpy as np
import pytest
from seeded_instances import seeded_instances


def test_weak_extension_and_unseeded_noise():
    p = np.zeros((32, 32), np.float32)
    p[4:8, 4:8] = .95
    p[8:12, 4:8] = .7
    p[22:26, 22:26] = .7
    masks = seeded_instances(p, np.ones_like(p, bool), min_seed=4, min_area=4)
    assert len(masks) == 1 and masks[0].sum() == 32


@pytest.mark.parametrize('separate, count', [(True, 2), (False, 1)])
def test_weak_bridge_and_support(separate, count):
    p = np.zeros((32, 32), np.float32)
    p[10:14, 3:8] = .95
    p[10:14, 20:25] = .95
    p[11:13, 8:20] = .7
    valid = np.ones_like(p, bool)
    valid[:, 24:] = False
    masks = seeded_instances(p, valid, min_seed=4, min_area=4, separate=separate)
    assert len(masks) == count
    assert np.stack(masks).sum(0).max() == 1
    assert not np.stack(masks)[:, ~valid].any()


def test_empty_and_invalid_inputs():
    p = np.zeros((8, 8))
    assert seeded_instances(p, np.ones_like(p, bool)) == []
    with pytest.raises(ValueError):
        seeded_instances(p, np.ones_like(p, bool), low=.9, high=.8)
