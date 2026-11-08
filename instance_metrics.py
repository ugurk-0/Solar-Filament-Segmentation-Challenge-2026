"""Instance PQ from cached IoU matrices; shared by retained YOLO experiments."""
import numpy as np


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
