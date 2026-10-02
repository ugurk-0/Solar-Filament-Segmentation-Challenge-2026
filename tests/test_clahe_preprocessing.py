from types import SimpleNamespace

import numpy as np
import pytest

from clahe_preprocessing import clahe_equalize


def test_clahe_preserves_range_shape_dtype_and_feature_location():
    y, x = np.ogrid[:192, :192]
    image = np.clip(0.15 + 0.7 * ((x + y) / (2 * 191)), 0, 1).astype(np.float32)
    image[88:104, 120:136] = 0.0
    result = clahe_equalize(image, clip_limit=0.01)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    assert np.unravel_index(np.argmin(result), result.shape) == np.unravel_index(np.argmin(image), image.shape)


def test_clahe_is_deterministic_for_identical_input():
    rng = np.random.default_rng(20261002)
    image = rng.uniform(0.05, 0.95, size=(128, 128)).astype(np.float32)
    first = clahe_equalize(image, clip_limit=0.01)
    second = clahe_equalize(image, clip_limit=0.01)
    assert np.array_equal(first, second)


def test_clahe_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        clahe_equalize(np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError):
        clahe_equalize(np.ones((8, 8), dtype=np.float32), clip_limit=0)


def test_dataset_applies_clahe_once_and_preserves_targets(monkeypatch):
    from experiment_clahe import ClaheDataset, OriginalDataset

    rng = np.random.default_rng(123)
    image = rng.uniform(0.1, 0.9, size=(96, 96)).astype(np.float32)
    mask = image < 0.25
    auxiliary = np.zeros_like(image)
    calls = []

    def raw_load(self, image_id):
        calls.append(image_id)
        return image, mask, auxiliary

    monkeypatch.setattr(OriginalDataset, "_load_full", raw_load)
    cfg = SimpleNamespace(clahe_enabled=True, clahe_kernel_size=None, clahe_clip_limit=0.01, clahe_nbins=256)
    dataset = ClaheDataset(None, [1], cfg, False, np.ones_like(mask))
    first = dataset._load_full(1)
    second = dataset._load_full(1)
    assert calls == [1] and first is second
    assert np.array_equal(first[0], clahe_equalize(image, clip_limit=0.01))
    assert first[1] is mask and first[2] is auxiliary