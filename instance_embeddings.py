"""Embedding-space instance separation utilities for filament masks."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def discriminative_embedding_loss(embeddings, labels, valid, delta_var=0.5,
                                  delta_dist=1.5, regularization=0.001):
    """Pull each instance toward its center and push centers apart."""
    if embeddings.ndim != 4 or labels.ndim != 3 or valid.ndim != 3:
        raise ValueError("Expected embeddings [B,D,H,W], labels/valid [B,H,W]")
    if embeddings.shape[0] != labels.shape[0] or embeddings.shape[-2:] != labels.shape[-2:]:
        raise ValueError("Embedding and label coordinates must match")
    if valid.shape != labels.shape:
        raise ValueError("Label and valid coordinates must match")
    total_var = embeddings.new_zeros(())
    total_dist = embeddings.new_zeros(())
    total_reg = embeddings.new_zeros(())
    counted = 0
    for batch in range(embeddings.shape[0]):
        pixels = embeddings[batch].permute(1, 2, 0)[valid[batch].bool()]
        instance_ids = labels[batch][valid[batch].bool()].long()
        ids = instance_ids.unique()
        ids = ids[ids > 0]
        if not len(ids):
            continue
        centers = []
        for instance_id in ids:
            points = pixels[instance_ids == instance_id]
            center = points.mean(0)
            centers.append(center)
            total_var = total_var + F.relu((points - center).norm(dim=1) - delta_var).square().mean()
        centers = torch.stack(centers)
        total_reg = total_reg + centers.norm(dim=1).mean()
        if len(centers) > 1:
            distances = torch.pdist(centers)
            total_dist = total_dist + F.relu(2 * delta_dist - distances).square().mean()
        counted += 1
    if not counted:
        return embeddings.square().sum() * 0
    return (total_var + total_dist + regularization * total_reg) / counted


def embedding_to_instances(embeddings, probability, limb, threshold=0.8,
                           min_area=500, eps=0.7, min_samples=8):
    """Cluster foreground pixels in embedding space and return binary masks."""
    if embeddings.ndim != 3 or probability.shape != embeddings.shape[1:]:
        raise ValueError("Embedding and probability coordinates must match")
    foreground = (probability > threshold) & limb
    coordinates = np.argwhere(foreground)
    if len(coordinates) < min_samples:
        return []
    values = embeddings[:, foreground].T.astype(np.float32, copy=False)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(values)
    instances = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        pixels = coordinates[labels == label]
        if len(pixels) < min_area:
            continue
        mask = np.zeros(probability.shape, dtype=bool)
        mask[pixels[:, 0], pixels[:, 1]] = True
        instances.append(mask)
    return instances