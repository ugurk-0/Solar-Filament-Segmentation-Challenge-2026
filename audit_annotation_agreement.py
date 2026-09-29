"""Measure annotation variability on sampled training observations only."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

import pq_experiment as ex
import solution as sol


def main():
    torch.set_num_threads(2)
    cfg = ex.config("runs/window_context_20260929")
    coco, train_ids, _ = sol.get_fold(cfg)
    groups = defaultdict(list)
    for image_id in train_ids:
        groups[sol.base_name(coco.imgs[image_id]["file_name"])].append(image_id)
    names = sorted(name for name, ids in groups.items() if len(ids) > 1)
    selected = np.random.default_rng(2026).choice(names, min(24, len(names)), replace=False).tolist()
    limb, rows = sol.build_limb_mask(cfg), []
    for name in selected:
        pair = groups[name][:2]
        masks = []
        for image_id in pair:
            info = coco.imgs[image_id]
            instances = [sol.polygon_to_mask(ann, info["height"], info["width"]) & limb
                         for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))]
            masks.append([mask for mask in instances if mask.any()])
        pq, sq, rq = sol.panoptic_quality(*masks)
        rows.append(dict(observation=name, image_ids=pair,
                         n_instances=list(map(len, masks)), pq=pq, sq=sq, rq=rq))
    report = dict(scope="24 sampled duplicate-annotation groups from training observations only; descriptive agreement, not a performance ceiling",
                  mean_pair_pq=float(np.mean([row["pq"] for row in rows])),
                  median_pair_pq=float(np.median([row["pq"] for row in rows])), rows=rows)
    Path("reports/training_annotation_agreement_20260929.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
