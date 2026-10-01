"""Use annotated centre lines as auxiliary training targets, never model inputs."""
from collections import defaultdict
import random
import numpy as np
from scipy import ndimage
from skimage.draw import line
import torch
import solution as sol


def spine_heatmap(annotations, foreground, valid, sigma=3.0):
    if foreground.shape != valid.shape or sigma <= 0:
        raise ValueError('Invalid support or spine width')
    h, w = foreground.shape
    centre = np.zeros((h, w), bool)
    for annotation in annotations:
        points = np.asarray(annotation.get('spine', []), dtype=np.float32)
        if points.size < 4 or points.size % 2 or not np.isfinite(points).all():
            raise ValueError('Invalid annotated spine')
        points = np.rint(points.reshape(-1, 2)).astype(int)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            rr, cc = line(y0, x0, y1, x1)
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            centre[rr[inside], cc[inside]] = True
    centre &= foreground & valid
    if not centre.any():
        return np.zeros((h, w), np.float32)
    distance = ndimage.distance_transform_edt(~centre)
    target = np.exp(-0.5 * (distance / sigma)**2)
    target[(distance > 3 * sigma) | ~foreground | ~valid] = 0
    return target.astype(np.float32)


class ObservationDataset(sol.FilamentDataset):
    """Equal observation frequency, random actual annotation set on each visit."""
    def __init__(self, coco, img_ids, cfg, train, limb):
        groups = defaultdict(list)
        for image_id in img_ids:
            groups[sol.base_name(coco.imgs[image_id]['file_name'])].append(image_id)
        self.annotation_sets = {members[0]: sorted(members) for members in groups.values()}
        super().__init__(coco, list(self.annotation_sets), cfg, train, limb)

    def _load_full(self, image_id):
        actual = random.choice(self.annotation_sets[image_id]) if self.train else image_id
        if actual in self._cache:
            return self._cache[actual]
        image, mask, auxiliary = super()._load_full(actual)
        if self.train and getattr(self.cfg, 'aux_spine_weight', 0) > 0:
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[actual]))
            auxiliary = spine_heatmap(annotations, mask, self.limb)
        self._cache[actual] = image, mask, auxiliary
        return self._cache[actual]


def auxiliary_loss(logits, target, valid):
    """Balanced positive/background MSE, avoiding domination by empty pixels."""
    if logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Auxiliary logits/target/support must share coordinates')
    positive = (target > .05).float() * valid
    negative = (target <= .05).float() * valid
    error = (logits.float().sigmoid() - target.float()).square()
    return .5 * ((error * positive).sum() / positive.sum().clamp_min(1)
                 + (error * negative).sum() / negative.sum().clamp_min(1))
