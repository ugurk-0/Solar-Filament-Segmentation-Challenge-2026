"""Skeleton-endpoint graph merging for fragmented filament predictions.

Public-recipe reproduction (Qiuwei V2 decoding): find skeleton endpoints of each
predicted instance, connect pairs that are close, directionally aligned with the
local skeleton tangent, and separated by a gap whose probability is high enough.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def skeleton_endpoints(mask):
    """Return endpoint coordinates (row, col) and the local tangent direction."""
    skeleton = skeletonize(mask)
    kernel = np.ones((3, 3), int)
    kernel[1, 1] = 0
    neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
    endpoints = np.argwhere(skeleton & (neighbors == 1))
    result = []
    for y, x in endpoints:
        ys, xs = np.where(skeleton[max(0, y - 8):y + 9, max(0, x - 8):x + 9])
        if len(ys) < 2:
            continue
        ys, xs = ys + max(0, y - 8), xs + max(0, x - 8)
        direction = np.array([y - ys.mean(), x - xs.mean()], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        result.append(((y, x), direction / norm))
    return result


def _gap_probability(probability, a, b):
    """Mean probability along the straight gap between two endpoints."""
    (y0, x0), (y1, x1) = a, b
    steps = int(max(abs(y1 - y0), abs(x1 - x0), 1))
    rows = np.linspace(y0, y1, steps + 1).astype(int)
    cols = np.linspace(x0, x1, steps + 1).astype(int)
    inside = ((rows >= 0) & (rows < probability.shape[0])
              & (cols >= 0) & (cols < probability.shape[1]))
    if not inside.any():
        return 0.0
    return float(probability[rows[inside], cols[inside]].mean())


def merge_instances(instances, probability, max_distance=50, min_cosine=0.8,
                    min_gap_probability=0.4):
    """Merge instances whose skeleton endpoints bridge a high-probability gap."""
    if len(instances) < 2:
        return list(instances)
    endpoints = [skeleton_endpoints(mask) for mask in instances]
    parent = list(range(len(instances)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if find(i) == find(j):
                continue
            for (point_i, direction_i) in endpoints[i]:
                for (point_j, direction_j) in endpoints[j]:
                    distance = float(np.hypot(point_i[0] - point_j[0], point_i[1] - point_j[1]))
                    if distance > max_distance or distance < 1e-6:
                        continue
                    gap = np.array([point_j[0] - point_i[0], point_j[1] - point_i[1]], dtype=np.float64)
                    gap /= np.linalg.norm(gap)
                    if abs(float(direction_i @ gap)) < min_cosine:
                        continue
                    if _gap_probability(probability, point_i, point_j) < min_gap_probability:
                        continue
                    parent[find(i)] = find(j)
    groups = {}
    for i in range(len(instances)):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(instances[members[0]])
        else:
            merged.append(np.logical_or.reduce([instances[k] for k in members]))
    return merged
