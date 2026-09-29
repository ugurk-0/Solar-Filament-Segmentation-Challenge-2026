"""Learn instance quality on top of the existing semantic U-Net.

Inspired by Mask Scoring R-CNN (Huang et al., CVPR 2019), not a reproduction.
Only competition training masks supervise this small, independently trained CNN.
"""
from dataclasses import replace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import solution as sol

PATCH_SIZE = 96
PROPOSALS = {"strict": {"thresh": 0.8, "min_area": 500},
             "permissive": {"thresh": 0.65, "min_area": 200}}
CUTOFFS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


class InstanceQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, incoming = [], 4
        for channels in (16, 32, 64, 96):
            blocks += [nn.Conv2d(incoming, channels, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(4, channels), nn.SiLU()]
            incoming = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(102, 64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 2))

    def forward(self, patches, geometry):
        return self.head(torch.cat([self.features(patches).flatten(1), geometry], dim=1))


def proposal_features(image, probability, masks, valid):
    """Aligned gray/probability/instance/support patches plus invariant geometry."""
    if image.shape != probability.shape or valid.shape != image.shape:
        raise ValueError("Image, probability, and support must share coordinates")
    patches, geometry = [], []
    h, w = image.shape
    for mask in masks:
        if mask.shape != image.shape or not mask.any() or np.any(mask & ~valid):
            raise ValueError("Proposal must be nonempty and confined to annotation support")
        ys, xs = np.where(mask)
        width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        side = max(96, int(1.5 * max(width, height)))
        cy, cx = int((ys.min() + ys.max()) // 2), int((xs.min() + xs.max()) // 2)
        y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
        y1, x1 = min(h, y0 + side), min(w, x0 + side)
        sl = np.s_[y0:y1, x0:x1]
        support = valid[sl]
        gray = image[sl].astype(np.float32)
        values = gray[support]
        mean, std = float(values.mean()), max(float(values.std()), 0.01)
        normalized = np.clip((gray - mean) / (4 * std), -1, 1) * support
        continuous = torch.from_numpy(np.stack([normalized, probability[sl] * support]))[None].float()
        discrete = torch.from_numpy(np.stack([mask[sl], support]).astype(np.float32))[None]
        patch = torch.cat([F.interpolate(continuous, (PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False),
                           F.interpolate(discrete, (PATCH_SIZE, PATCH_SIZE), mode="nearest")], dim=1)[0]
        # Interpolation must not introduce nonzero features outside support.
        patch[:3] *= patch[3:4]
        p = probability[mask]
        geometry.append([np.log1p(mask.sum()) / 12, float(mask.sum()) / (width * height),
                         np.log(max(width, height) / min(width, height)) / 5,
                         float(p.mean()), float(p.std()),
                         np.clip((float(image[mask].mean()) - mean) / (4 * std), -1, 1)])
        patches.append(patch.numpy())
    return (np.stack(patches).astype(np.float16) if patches else np.empty((0, 4, PATCH_SIZE, PATCH_SIZE), np.float16),
            np.asarray(geometry, dtype=np.float32).reshape(-1, 6))


def matrix_metrics(iou, keep=None):
    """Same greedy matching as solution.match_instances, on cached float64 IoUs."""
    if keep is not None:
        iou = iou[keep]
    pred, gt = iou.shape
    used_p, used_g, values = set(), set(), []
    for flat in np.argsort(iou.ravel())[::-1]:
        value = float(iou.ravel()[flat])
        if value < 0.5:
            break
        p, g = np.unravel_index(flat, iou.shape)
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            values.append(value)
    tp, fp, fn = len(values), pred - len(values), gt - len(values)
    denominator = tp + 0.5 * (fp + fn)
    return dict(pq=sum(values) / denominator if denominator else 0.0,
                tp=tp, fp=fp, fn=fn, matched_iou=values)


def summarize(rows):
    tp, fp, fn = [sum(row[key] for row in rows) for key in ("tp", "fp", "fn")]
    total = sum(sum(row["matched_iou"]) for row in rows)
    denominator = tp + 0.5 * (fp + fn)
    return dict(n=len(rows), mean_pq=float(np.mean([r["pq"] for r in rows])),
                dataset_pq=total / denominator if denominator else 0.0,
                sq=total / tp if tp else 0.0, rq=tp / denominator if denominator else 0.0,
                tp=tp, fp=fp, fn=fn)


@torch.no_grad()
def quality_scores(model, patches, geometry, device, batch_size=64):
    model.eval()
    result = []
    for start in range(0, len(patches), batch_size):
        output = model(torch.from_numpy(patches[start:start + batch_size].astype(np.float32)).to(device),
                       torch.from_numpy(geometry[start:start + batch_size]).to(device)).sigmoid()
        result.extend((output[:, 0] * output[:, 1]).cpu().tolist())
    return np.asarray(result, dtype=np.float32)


def filter_instances(model, image, probability, masks, valid, cutoff, device):
    if cutoff <= 0 or not masks:
        return masks
    patches, geometry = proposal_features(image, probability, masks, valid)
    scores = quality_scores(model, patches, geometry, device)
    return [mask for mask, score in zip(masks, scores) if score >= cutoff]
