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
