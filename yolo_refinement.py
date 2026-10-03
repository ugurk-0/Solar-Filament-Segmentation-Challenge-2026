"""Use a retained U-Net probability map to refine YOLO instance boundaries."""
import cv2
import numpy as np
from yolo_data import disjoint_masks


def refine_masks(masks, scores, probability, valid, radius=8, threshold=.5,
                 confidence=.1, min_area=128):
    if probability.shape != valid.shape or not np.isfinite(probability).all():
        raise ValueError('Probability and support must be finite native-coordinate maps')
    if radius < 0:
        raise ValueError('Radius must be nonnegative')
    refined, kept_scores = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for mask, score in zip(masks, scores):
        if score < confidence:
            continue
        mask = np.asarray(mask, bool)
        if mask.shape != valid.shape:
            raise ValueError('YOLO masks must use native coordinates')
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        y0, y1 = max(0, int(ys.min()) - radius), min(mask.shape[0], int(ys.max()) + radius + 1)
        x0, x1 = max(0, int(xs.min()) - radius), min(mask.shape[1], int(xs.max()) + radius + 1)
        local = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), kernel).astype(bool)
        local &= (probability[y0:y1, x0:x1] >= threshold) & valid[y0:y1, x0:x1]
        result = np.zeros_like(valid, bool)
        result[y0:y1, x0:x1] = local
        refined.append(result)
        kept_scores.append(score)
    return disjoint_masks(refined, kept_scores, valid, confidence=confidence, min_area=min_area)
