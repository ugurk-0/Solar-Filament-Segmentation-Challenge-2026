import numpy as np
import pytest
import torch

from instance_quality import InstanceQualityNet, proposal_features, matrix_metrics, filter_instances
from solution import instance_iou_matrix, panoptic_quality


@pytest.mark.parametrize("case", ["empty", "one", "merge", "split", "below"])
def test_cached_metric_matches_pipeline(case):
    a, b = np.zeros((12, 12), bool), np.zeros((12, 12), bool)
    a[1:5, 1:5] = True
    b[7:11, 7:11] = True
    split = a.copy()
    split[3:] = False
    pred, gt = {"empty": ([], []), "one": ([a], [a]), "merge": ([a | b], [a, b]),
                "split": ([split, a & ~split], [a]), "below": ([b], [a])}[case]
    actual = matrix_metrics(instance_iou_matrix(pred, gt))
    assert actual["pq"] == pytest.approx(panoptic_quality(pred, gt)[0])


def test_features_preserve_support_and_network_has_finite_gradients():
    image = np.ones((128, 128), np.float32) * 0.6
    mask = np.zeros_like(image, dtype=bool)
    mask[40:50, 40:60] = True
    image[mask] = 0.2
    probability = mask.astype(np.float32) * 0.9
    valid = np.ones_like(mask)
    valid[:35] = False
    patches, geometry = proposal_features(image, probability, [mask], valid)
    assert patches.shape == (1, 4, 96, 96) and geometry.shape == (1, 6)
    assert np.isfinite(patches).all() and np.isfinite(geometry).all()
    assert not patches[0, :3, patches[0, 3] == 0].any()
    net = InstanceQualityNet()
    prediction = net(torch.from_numpy(patches).float(), torch.from_numpy(geometry))
    assert prediction.shape == (1, 2) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    masks = [mask]
    assert filter_instances(net, image, probability, masks, valid, 0, "cpu") is masks


def test_features_reject_masks_outside_annotation_support():
    image = np.zeros((16, 16), np.float32)
    mask = np.ones_like(image, dtype=bool)
    with pytest.raises(ValueError, match="annotation support"):
        proposal_features(image, image, [mask], np.zeros_like(mask))
