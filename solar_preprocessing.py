"""Bounded radial illumination correction for centred full-disk images."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def radial_correct(image, radius=950, strength=0.5):
    """Estimate quiet-Sun background with annular medians; preserve coordinates.

    This is illumination correction, not geometric deprojection or denoising.
    Gain is bounded to avoid amplifying dark/cloud-contaminated annuli excessively.
    """
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Expected a finite grayscale image')
    if radius <= 0 or not 0 <= strength <= 1:
        raise ValueError('Invalid radius or correction strength')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('Expected intensities in [0, 1]')
    if strength == 0:
        return image.copy()
    h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((xx - w // 2)**2 + (yy - h // 2)**2) / radius
    # Subsample only for robust background estimation, never the returned image.
    sample_r, sample = r[::4, ::4], image[::4, ::4]
    bins = np.minimum((sample_r * 64).astype(int), 63)
    medians = np.full(64, np.nan)
    for k in range(64):
        values = sample[(bins == k) & (sample_r < .98)]
        if len(values):
            medians[k] = np.median(values)
    good = np.isfinite(medians)
    if good.sum() < 4:
        raise ValueError('Insufficient disk samples for radial estimation')
    profile = np.interp(np.arange(64), np.where(good)[0], medians[good])
    profile = gaussian_filter1d(profile, 2, mode='nearest')
    reference = float(np.median(profile[:20]))
    if reference < .01:
        return image.copy()
    background = np.interp(r, (np.arange(64) + .5) / 64, profile)
    gain = np.clip(reference / np.maximum(background, .05), .75, 1.8)
    gain = np.where(r <= 1, 1 + strength * (gain - 1), 1)
    return np.clip(image * gain, 0, 1).astype(np.float32)
