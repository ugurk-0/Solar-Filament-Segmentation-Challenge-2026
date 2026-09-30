"""Grow confident components into weaker foreground while preserving support."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

import solution as sol


def seeded_instances(probability, valid, low=0.65, high=0.8, min_seed=500,
                     min_area=500, separate=True):
    if not 0 <= low <= high <= 1:
        raise ValueError('Require 0 <= low <= high <= 1')
    if probability.shape != valid.shape or not np.isfinite(probability).all():
        raise ValueError('Probability and support must be finite and aligned')
    if min_seed < 1 or min_area < 1:
        raise ValueError('Areas must be positive')
    seeds, _ = ndimage.label((probability > high) & valid)
    sizes = np.bincount(seeds.ravel())
    keep = sizes >= min_seed
    keep[0] = False
    seeds = np.where(keep[seeds], seeds, 0)
    if not seeds.any():
        return []
    foreground = (probability > low) & valid
    if separate:
        labels = watershed(-probability, seeds, mask=foreground)
    else:
        labels, _ = ndimage.label(foreground)
        accepted = np.unique(labels[seeds > 0])
        labels = np.where(np.isin(labels, accepted), labels, 0)
    masks = []
    occupied = np.zeros_like(valid, dtype=bool)
    for label, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        original = labels[box] == label
        cleaned = sol.postprocess_one(original) & valid[box]
        if np.any(cleaned & (((labels[box] != 0) & (labels[box] != label)) | occupied[box])):
            cleaned = original
        if cleaned.sum() >= min_area:
            mask = np.zeros_like(valid, dtype=bool)
            mask[box] = cleaned
            occupied[box] |= cleaned
            masks.append(mask)
    return masks
