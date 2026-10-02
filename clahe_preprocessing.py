"""Controlled CLAHE reproduction for grayscale solar filament images."""

import numpy as np
from skimage import exposure


def clahe_equalize(image, kernel_size=None, clip_limit=0.01, nbins=256):
    """Apply CLAHE while preserving the image grid, shape, and [0, 1] range."""
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite grayscale image")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Expected intensities in [0, 1]")
    if clip_limit <= 0 or nbins < 2:
        raise ValueError("Invalid CLAHE parameters")
    result = exposure.equalize_adapthist(
        image,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins,
    )
    if result.shape != image.shape:
        raise ValueError("CLAHE altered image shape")
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError("CLAHE produced values outside [0, 1]")
    return result.astype(np.float32, copy=False)