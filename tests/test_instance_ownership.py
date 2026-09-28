import numpy as np

from solution import Cfg, prob_to_instances


def test_hole_filling_preserves_nested_instance_identity():
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[3:10, 3:10] = 0.9
    probability[4:9, 4:9] = 0
    probability[6, 6] = 0.9
    cfg = Cfg(instance_method="components", thresh=0.8, min_area=1, close_kernel=1)
    masks = prob_to_instances(probability, cfg, np.ones_like(probability, dtype=bool))
    assert len(masks) == 2
    assert not np.any(masks[0] & masks[1])
    assert sorted(int(m.sum()) for m in masks) == [1, 24]
    assert np.array_equal(np.logical_or.reduce(masks), probability > 0.8)


def test_unoccupied_small_hole_still_filled():
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[3:10, 3:10] = 0.9
    probability[6, 6] = 0
    cfg = Cfg(instance_method="components", thresh=0.8, min_area=1, close_kernel=1)
    masks = prob_to_instances(probability, cfg, np.ones_like(probability, dtype=bool))
    assert len(masks) == 1
    assert masks[0].sum() == 49
