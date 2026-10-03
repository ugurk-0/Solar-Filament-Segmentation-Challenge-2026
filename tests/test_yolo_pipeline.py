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


def test_chunked_prediction_matches_upstream_native_masks():
    pytest.importorskip('ultralytics')
    import torch
    from types import SimpleNamespace
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from yolo_prediction import ChunkedSegmentationPredictor
    predictor = object.__new__(ChunkedSegmentationPredictor)
    predictor.args = SimpleNamespace(retina_masks=True)
    predictor.model = SimpleNamespace(names={0: 'filament'})
    generator = torch.Generator().manual_seed(7)
    proto = torch.randn((4, 16, 16), generator=generator)
    pred = torch.zeros((17, 10))
    pred[:, :4] = torch.tensor([4., 8., 60., 56.])
    pred[:, 4] = .8
    pred[:, 6:] = torch.randn((17, 4), generator=generator)
    img, original = torch.zeros((1, 3, 64, 64)), np.zeros((128, 128, 3), np.uint8)
    expected = SegmentationPredictor.construct_result(predictor, pred.clone(), img, original, 'test.png', proto)
    actual = predictor.construct_result(pred.clone(), img, original, 'test.png', proto)
    assert torch.equal(expected.boxes.data, actual.boxes.data)
    assert torch.equal(expected.masks.data, actual.masks.data)
