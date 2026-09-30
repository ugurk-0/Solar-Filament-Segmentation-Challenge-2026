import numpy as np
import pytest
from solar_preprocessing import radial_correct


def test_radial_correction_reduces_gradient_preserves_dark_feature_and_exterior():
    y, x = np.ogrid[:256, :256]
    r = np.sqrt((x-128)**2 + (y-128)**2) / 110
    image = np.where(r <= 1, .6 - .25*r*r, .03).astype(np.float32)
    image[120:124, 190:208] *= .5
    result = radial_correct(image, 110, .5)
    center = r < .2
    edge = (r > .8) & (r < .9)
    assert np.median(result[center]) - np.median(result[edge]) < np.median(image[center]) - np.median(image[edge])
    assert result[121, 200] < .6 * result[125, 200]
    assert np.array_equal(result[r > 1], image[r > 1])
    assert result.dtype == np.float32 and result.shape == image.shape
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1


def test_identity_and_black_images():
    image = np.zeros((128, 128), np.float32)
    assert np.array_equal(radial_correct(image, 50), image)
    image[:] = .5
    assert np.array_equal(radial_correct(image, 50, 0), image)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        radial_correct(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        radial_correct(np.zeros((128, 128)), strength=2)


def test_dataset_corrects_once_and_preserves_targets(monkeypatch):
    from types import SimpleNamespace
    from experiment_solar_preprocessing import SolarDataset, OriginalDataset
    y, x = np.ogrid[:128, :128]
    image = np.maximum(.1, .6 - .3*((x-64)**2 + (y-64)**2)/50**2).astype(np.float32)
    mask = image < .3
    auxiliary = np.zeros_like(image)
    calls = []
    def raw_load(self, image_id):
        calls.append(image_id)
        return image, mask, auxiliary
    monkeypatch.setattr(OriginalDataset, '_load_full', raw_load)
    cfg = SimpleNamespace(radial_strength=.5, disk_radius=50)
    dataset = SolarDataset(None, [1], cfg, False, np.ones_like(mask))
    first = dataset._load_full(1)
    second = dataset._load_full(1)
    assert calls == [1] and first is second
    assert np.array_equal(first[0], radial_correct(image, 50, .5))
    assert first[1] is mask and first[2] is auxiliary
