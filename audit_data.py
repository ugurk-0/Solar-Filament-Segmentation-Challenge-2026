"""Training-only annotation support audit; does not choose parameters from test data."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from solution import get_fold, build_limb_mask, polygon_to_mask, prob_to_instances, panoptic_quality
from pq_experiment import config


def main():
    cfg = config("runs/pq_research_20260926")
    coco, train_ids, val_ids = get_fold(cfg)
    old = build_limb_mask(replace(cfg, limb_geometry="legacy_wedge"))
    new = build_limb_mask(cfg)
    selected = np.random.default_rng(2026).choice(train_ids, 48, replace=False).tolist()
    areas = []; retained_old = []; retained_new = []; oracle = []
    for index, i in enumerate(selected):
        info = coco.imgs[i]
        gt = [polygon_to_mask(a, info["height"], info["width"])
              for a in coco.loadAnns(coco.getAnnIds(imgIds=[i]))]
        for mask in gt:
            areas.append(int(mask.sum()))
            retained_old.append(int((mask & old).sum()))
            retained_new.append(int((mask & new).sum()))
        masked = [mask & new for mask in gt if (mask & new).any()]
        if masked:
            probability = np.logical_or.reduce(masked).astype(np.float32)
            scores = {method: panoptic_quality(prob_to_instances(probability, replace(cfg, instance_method=method), new), masked)[0]
                      for method in ("components", "watershed")}
            oracle.append(scores)
        print(f"support audit {index+1}/{len(selected)}", flush=True)
    result = {"sample": "48 training annotation records; no held-out or test labels used", "image_ids": selected,
              "n_instances": len(areas), "old_support_fraction": float(old.mean()),
              "new_support_fraction": float(new.mean()),
              "old_retained_annotation_pixels": float(sum(retained_old) / sum(areas)),
              "new_retained_annotation_pixels": float(sum(retained_new) / sum(areas)),
              "annotation_area_quantiles": dict(zip(["p05", "p50", "p95"], np.quantile(areas, [.05, .5, .95]).tolist())),
              "perfect_union_postprocess_mean_pq": {m: float(np.mean([r[m] for r in oracle])) for m in ("components", "watershed")},
              "geometry_assumption": "Orthographic solar projection B0=0, fixed radius 950; approximate annotation support, not per-image WCS."}
    Path(cfg.work_dir, "data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k,v in result.items() if k != "image_ids"}, indent=2))


if __name__ == "__main__":
    main()
