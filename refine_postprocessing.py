"""Calibrate Round 5 instance extraction without using test images or labels.

Runs from the repository root; preserves the previous submission. Uses the fixed
40-image calibration split, freezes a setting, then assesses it on 85 images.
"""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import pq_experiment as experiment
import solution as sol


ROOT = Path("runs/pq_research_20260926")
OUT = Path("runs/pq_refinement_20260928_fixed")
CHECKPOINT = ROOT / "training_round5/fold1_best.pt"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = experiment.config(OUT, fold=1)
    coco, _, ids = sol.get_fold(cfg)
    splits = experiment.partition(ids, cfg.seed)
    _, ds = sol.make_loader(coco, ids, cfg, False)
    with CHECKPOINT.open("rb") as stream:
        signature = hashlib.file_digest(stream, "sha256").hexdigest()
    old_dir = ROOT / "comparison_round5"
    old_protocol = json.loads((old_dir / "protocol.json").read_text())
    assert old_protocol["checkpoint_sha256"] == signature
    for key in ("tta", "window", "overlap", "img_size", "limb_geometry", "limb_max_deg", "disk_radius"):
        assert old_protocol["config"][key] == getattr(cfg, key), key
    cache = old_dir / ("probabilities_" + signature[:12])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
    model.load_state_dict(state["ema"])
    model.eval()
    del state
    settings = [dict(instance_method="components", thresh=t, min_area=a, close_kernel=c)
                for t in (0.5, 0.65, 0.8, 0.9) for c in (1, 3) for a in (200, 500)]
    sol._write_json_atomically(OUT / "protocol.json", {
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": signature,
        "config": vars(cfg), "splits": splits, "candidates": settings,
        "hypothesis": "Stricter confidence and size filters reduce false positives; removing closing may avoid merging nearby filaments.",
        "selection": "40 calibration observations only; 85-observation assessment reused from prior research",
    })

    def probability(i):
        path = cache / f"{i}.npy"
        if not path.exists():
            image, _, _ = ds._load_full(i)
            pred = sol.predict_full([model], image, cfg)
            assert np.isfinite(pred).all() and pred.shape == ds.limb.shape
            np.save(path, pred)
        return np.load(path)

    def row(pred, gt, i):
        _, matches = sol.match_instances(pred, gt)
        tp = len(matches)
        fp, fn = len(pred) - tp, len(gt) - tp
        matched = [m[2] for m in matches]
        denom = tp + (fp + fn) / 2
        return dict(image_id=i, file_name=coco.imgs[i]["file_name"],
                    pq=sum(matched) / denom if denom else 0,
                    tp=tp, fp=fp, fn=fn, matched_iou=matched)

    def summary(rows):
        tp, fp, fn = [sum(r[k] for r in rows) for k in ("tp", "fp", "fn")]
        ious = sum(sum(r["matched_iou"]) for r in rows)
        denom = tp + (fp + fn) / 2
        return dict(n=len(rows), mean_pq=float(np.mean([r["pq"] for r in rows])),
                    dataset_pq=ious / denom if denom else 0,
                    sq=ious / tp if tp else 0, rq=tp / denom if denom else 0,
                    tp=tp, fp=fp, fn=fn)

    candidate_rows = [[] for _ in settings]
    for index, i in enumerate(splits["calibration"], 1):
        p, gt = probability(i), sol.ground_truth_instances(ds, i)
        for k, setting in enumerate(settings):
            candidate_rows[k].append(row(sol.prob_to_instances(p, replace(cfg, **setting), ds.limb), gt, i))
        print(f"Calibration {index}/40", flush=True)
    grid = [{"parameters": s, "summary": summary(r)} for s, r in zip(settings, candidate_rows)]
    sol._write_json_atomically(OUT / "calibration_grid.json", grid)
    best = max(grid, key=lambda r: r["summary"]["mean_pq"])
    sol._write_json_atomically(OUT / "best_postproc.json", best)
    print("Frozen calibration winner:", best, flush=True)
    after, before = [], []
    baseline_cfg = replace(cfg, instance_method="components", thresh=0.65,
                           min_area=200, close_kernel=3)
    for index, i in enumerate(splits["audit"], 1):
        p, gt = probability(i), sol.ground_truth_instances(ds, i)
        pred = sol.prob_to_instances(p, replace(cfg, **best["parameters"]), ds.limb)
        after.append(row(pred, gt, i))
        before.append(row(sol.prob_to_instances(p, baseline_cfg, ds.limb), gt, i))
        if index % 10 == 0:
            print(f"Assessment {index}/85", flush=True)
    result = {"before": summary(before), "after": summary(after),
              "paired": experiment.paired_bootstrap(before, after),
              "parameters": best["parameters"], "selected_rows": after}
    sol._write_json_atomically(OUT / "assessment.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected_rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
