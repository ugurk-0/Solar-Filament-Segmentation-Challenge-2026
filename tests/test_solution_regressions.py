import numpy as np
import torch

from solution import (Cfg, build_limb_mask, dihedral_tta_pairs, mask_to_rle,
                      maskutils, panoptic_quality, predict_full,
                      prob_to_instances, soft_cldice, EMA, evaluate_detailed,
                      FilamentDataset)
def square(top, left, size=2, shape=(8, 8)):
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + size, left:left + size] = True
    return mask


def test_panoptic_quality_edge_cases():
    assert panoptic_quality([], []) == (0.0, 0.0, 0.0)
    one = square(1, 1)
    assert panoptic_quality([one], [one]) == (1.0, 1.0, 1.0)

    far_apart = square(5, 5)
    merged = np.zeros((8, 8), dtype=bool)
    merged[1:7, 1:7] = True
    assert panoptic_quality([merged], [one, far_apart])[0] == 0.0


def test_mask_rle_round_trip():
    mask = square(2, 3)
    encoded = mask_to_rle(mask)
    decoded = maskutils.decode({"size": list(mask.shape), "counts": encoded})
    assert np.array_equal(decoded.astype(bool), mask)


def test_limb_mask_keeps_meridian_and_excludes_far_limb():
    cfg = Cfg(img_size=128, disk_radius=60, limb_max_deg=70)
    mask = build_limb_mask(cfg)
    center = cfg.img_size // 2
    assert mask[center, center]
    assert mask.mean() > 0.5
    assert not mask[center, center + cfg.disk_radius - 2]
    assert mask[center, center + 20]  # equator is not an excluded wedge
    assert np.array_equal(mask[40:89, 40:89], mask[40:89, 40:89][::-1, ::-1])


def test_dihedral_tta_pairs_restore_original_tensor():
    image = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    for augment, invert in dihedral_tta_pairs(enabled=True):
        assert torch.equal(invert(augment(image)), image)


def test_prob_to_instances_respects_threshold_and_components():
    cfg = Cfg(thresh=0.5, min_area=1, min_dist_ws=10, close_kernel=1)
    limb = np.ones((16, 16), dtype=bool)
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[2:4, 2:4] = 0.5
    assert prob_to_instances(probability, cfg, limb) == []

    probability[2:4, 2:4] = 0.6
    probability[10:12, 10:12] = 0.6
    instances = prob_to_instances(probability, cfg, limb)
    assert len(instances) == 2
    assert sorted(mask.sum() for mask in instances) == [4, 4]


def test_batched_tta_matches_single_view_inference():
    class IdentitySegmenter(torch.nn.Module):
        def forward(self, image):
            return image, image

    image = np.arange(36, dtype=np.float32).reshape(6, 6) / 36
    model = IdentitySegmenter()
    single_view_cfg = Cfg(device="cpu", window=4, overlap=2, tta=True, tta_batch_size=1)
    batched_cfg = Cfg(device="cpu", window=4, overlap=2, tta=True, tta_batch_size=2)
    np.testing.assert_allclose(
        predict_full([model], image, single_view_cfg),
        predict_full([model], image, batched_cfg),
        rtol=0,
        atol=1e-7,
    )


def test_pq_split_below_threshold_and_empty_sides():
    gt = np.zeros((8, 8), bool); gt[2:6, 2:6] = True
    left = gt.copy(); left[:, 4:] = False
    right = gt & ~left
    # Exact 0.5 is accepted; one of the halves remains a false positive.
    np.testing.assert_allclose(panoptic_quality([left, right], [gt]), (1/3, 0.5, 2/3))
    small = np.zeros_like(gt); small[2:3, 2:6] = True
    assert panoptic_quality([small], [gt]) == (0, 0, 0)
    assert panoptic_quality([], [gt]) == (0, 0, 0)
    assert panoptic_quality([gt], []) == (0, 0, 0)


def test_cldice_identity_break_and_masked_gradient():
    target = torch.zeros(1, 1, 32, 32); target[:, :, 16, 3:29] = 1
    valid = torch.ones_like(target); valid[:, :, :4] = 0
    assert soft_cldice(target, target, valid).item() == 0
    broken = target.clone(); broken[:, :, 16, 12:20] = 0
    assert soft_cldice(broken, target, valid).item() > 0
    pred = (target * 0.8 + 0.1).requires_grad_()
    loss = soft_cldice(pred, target, valid)
    assert 0 <= loss.item() <= 1
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, :, :4].abs().sum() == 0
    empty = torch.zeros_like(target)
    assert soft_cldice(empty, empty, valid).item() == 0


def test_ema_evaluation_preserves_live_parameters():
    model = torch.nn.Linear(1, 1)
    ema = EMA(model, 0.9)
    with torch.no_grad(): model.weight.add_(2)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    class EmptyValidation:
        ids = []
    evaluate_detailed(model, EmptyValidation(), Cfg(), ema)
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())


def test_long_filament_not_split_by_components_and_support_preserved():
    prob = np.zeros((64, 128), np.float32); prob[25:32, 5:120] = 1
    limb = np.ones_like(prob, bool); limb[:, 80:] = False
    cfg = Cfg(min_area=1, instance_method="components")
    result = prob_to_instances(prob, cfg, limb)
    assert len(result) == 1
    assert not result[0][:, 80:].any()


def test_watershed_retains_nearby_disconnected_objects():
    prob = np.zeros((32, 32), np.float32)
    prob[10:12, 10:12] = 1; prob[10:12, 15:17] = 1
    cfg = Cfg(min_area=1, min_dist_ws=25, close_kernel=1, instance_method="watershed")
    assert len(prob_to_instances(prob, cfg, np.ones_like(prob, bool))) == 2


def test_positive_crop_uses_paired_coordinates(monkeypatch):
    cfg = Cfg(img_size=64, crop_size=8, oversample_p=1)
    ds = FilamentDataset(None, [1], cfg, True, np.ones((64, 64), bool))
    mask = np.zeros((64, 64), bool); mask[12, 48] = 1; mask[48, 12] = 1
    ds._cache[1] = (mask.astype(np.float32), mask, np.zeros((64, 64), np.float32))
    for _ in range(10):
        x, y, _, vm = ds[0]
        assert y.sum() == 1
        assert torch.equal(x > 0.5, y > 0.5)
