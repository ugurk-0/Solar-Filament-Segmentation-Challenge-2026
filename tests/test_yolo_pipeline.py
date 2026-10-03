import numpy as np
import pytest
from yolo_data import mask_polygon, disjoint_masks


def test_polygon_roundtrip():
    mask = np.zeros((64, 64), bool)
    mask[12:16, 5:55] = True
    polygon, iou = mask_polygon(mask)
    assert polygon.shape[1] == 2 and iou == 1


def test_instance_ownership_support_and_small_masks():
    a = np.zeros((32, 32), bool)
    a[2:12, 2:12] = True
    b = np.zeros_like(a)
    b[7:17, 7:17] = True
    valid = np.ones_like(a)
    valid[:, :4] = False
    masks = disjoint_masks([a, b], [.9, .8], valid, min_area=4)
    assert len(masks) == 2
    assert not (masks[0] & masks[1]).any()
    assert not np.stack(masks)[:, ~valid].any()
    assert disjoint_masks([a], [.05], valid, confidence=.1) == []


def test_support_loss_has_zero_gradient_outside_annotation_region():
    pytest.importorskip('ultralytics')
    import torch
    from yolo_training import SupportSegmentationLoss, support_tensor
    proto = torch.zeros((1, 64, 64), requires_grad=True)
    target = torch.zeros((1, 64, 64))
    loss = SupportSegmentationLoss.single_mask_loss(target, torch.ones((1, 1)), proto,
                                                    torch.tensor([[0., 0., 64., 64.]]), torch.ones(1))
    loss.backward()
    valid = support_tensor(64, 'cpu').bool()
    assert not proto.grad[0, ~valid].any()
    assert proto.grad[0, valid].abs().sum() > 0
