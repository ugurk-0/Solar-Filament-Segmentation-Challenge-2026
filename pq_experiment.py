"""Reproducible PQ audit, calibration and paired comparison. No test-set tuning.

Run from the repository root. Probability caches are isolated by checkpoint
signature. The monitoring split agrees with train(eval_max_images=16).
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from solution import (Cfg, FilamentUNet, get_fold, make_loader, predict_full,
                      prob_to_instances, ground_truth_instances, image_metrics,
                      build_limb_mask, save_training_figures, _write_json_atomically)


def config(work_dir, fold=1):
    data = Path("MAGFiLO_1.0_Kaggle_2026/train").resolve()
    return Cfg(data_dir=str(data), train_json=str(data / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
               work_dir=str(work_dir), fold=fold, tta=False, window=1024,
               instance_method="components", preview_count=3)


def partition(ids, seed=2026):
    monitor = sorted(np.random.default_rng(seed).choice(ids, min(16, len(ids)), replace=False).tolist())
    rest = [i for i in ids if i not in monitor]
    rest = np.random.default_rng(seed + 1).permutation(rest).tolist()
    return {"monitor": monitor, "calibration": sorted(rest[:40]), "audit": sorted(rest[40:])}


def summarize(rows):
    tp, fp, fn = [sum(r[k] for r in rows) for k in ("tp", "fp", "fn")]
    ious = sum(sum(r["matched_iou"]) for r in rows)
    denom = tp + 0.5 * (fp + fn)
    return {"n": len(rows), "mean_pq": float(np.mean([r["pq"] for r in rows])),
            "dataset_pq": ious / denom if denom else 0,
            "sq": ious / tp if tp else 0, "rq": tp / denom if denom else 0,
            "tp": tp, "fp": fp, "fn": fn,
            "pixel_dice": float(np.mean([r["pixel_dice"] for r in rows]))}


def paired_bootstrap(before, after, seed=2026):
    a = {r["image_id"]: r for r in before}
    b = {r["image_id"]: r for r in after}
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical observation IDs")
    keys = sorted(a)
    diff = np.array([b[i]["pq"] - a[i]["pq"] for i in keys])
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(len(diff), size=(10000, len(diff)))].mean(axis=1)
    # Dates may remain temporally correlated even after exact-image grouping.
    dates = sorted({a[i]["file_name"].split("-")[-1][:8] for i in keys})
    blocks = [np.array([j for j, i in enumerate(keys) if a[i]["file_name"].split("-")[-1][:8] == d])
              for d in dates]
    block_samples = [diff[np.concatenate([blocks[k] for k in rng.integers(len(blocks), size=len(blocks))])].mean()
                     for _ in range(3000)]
    return {"mean_pq_delta": float(diff.mean()),
            "observation_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
            "date_block_bootstrap_95ci": np.quantile(block_samples, [0.025, 0.975]).tolist(),
            "n_observations": len(keys), "n_dates": len(dates),
            "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."}


def run(checkpoint, output, mode, baseline=None, parameters=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = config(output, state["cfg"]["fold"])
    coco, train_ids, ids = get_fold(cfg)
    splits = partition(ids, cfg.seed)
    groups = lambda selected: {coco.imgs[i]["file_name"].split("-")[-1] for i in selected}
    assert not groups(train_ids) & groups(ids)
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    protocol = {"split": splits, "config": vars(cfg), "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "hypothesis": "Avoiding EDT oversegmentation raises RQ and PQ; repaired training improves this further.",
                "primary_estimand": "mean per-physical-observation PQ with masked GT, IoU >= 0.5",
                "selection": "epoch: monitor; postprocessing: calibration; paired comparison: audit",
                "caveat": "Exploratory local experiment: prior agents already inspected aggregate fold-1 metrics."}
    signature = protocol["checkpoint_sha256"]
    cache = output / ("probabilities_" + signature[:12]); cache.mkdir(exist_ok=True)
    _write_json_atomically(output / "protocol.json", protocol)
    _, ds = make_loader(coco, ids, cfg, False)
    model = FilamentUNet(cfg, pretrained=False).to(cfg.device)
    model.load_state_dict(state["ema"]); del state
    model.eval()

    def probability(i):
        path = cache / f"{i}.npy"
        if path.exists(): return np.load(path).astype(np.float32)
        img, _, _ = ds._load_full(i)
        prob = predict_full([model], img, cfg)
        assert prob.shape == ds.limb.shape and np.isfinite(prob).all()
        # Float32 cache: calibration must see the exact inference probabilities.
        np.save(path, prob)
        return prob

    def score(selected, setting):
        current = replace(cfg, **setting)
        rows = []
        for j, i in enumerate(selected, 1):
            started = time.perf_counter()
            pred = prob_to_instances(probability(i), current, ds.limb)
            row = image_metrics(pred, ground_truth_instances(ds, i))
            row.update(image_id=i, file_name=coco.imgs[i]["file_name"])
            rows.append(row)
            print(f"[{mode}] {j}/{len(selected)} id={i} PQ={row['pq']:.4f} TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']} s={time.perf_counter()-started:.1f}", flush=True)
        return rows

    default = {"instance_method": "watershed", "thresh": 0.5, "min_area": 50}
    if mode == "baseline":
        grid = []
        for method in ("watershed", "components"):
            for threshold in (0.35, 0.5, 0.65):
                for area in (50, 200):
                    setting = {"instance_method": method, "thresh": threshold, "min_area": area}
                    rows = score(splits["calibration"], setting)
                    grid.append({"parameters": setting, "summary": summarize(rows)})
                    _write_json_atomically(output / "calibration_grid.json", grid)
        best = max(grid, key=lambda r: r["summary"]["mean_pq"])
        _write_json_atomically(output / "selected_postproc.json", best)
        chosen = best["parameters"]
        before = score(splits["audit"], default)
        after = score(splits["audit"], chosen)
        report = {"baseline": summarize(before), "selected": summarize(after),
                  "paired": paired_bootstrap(before, after), "parameters": chosen,
                  "baseline_rows": before, "selected_rows": after}
    else:
        chosen = json.loads(Path(parameters).read_text())["parameters"]
        after = score(splits["audit"], chosen)
        old = json.loads(Path(baseline).read_text())
        report = {"selected": summarize(after), "parameters": chosen,
                  "paired_training_effect": paired_bootstrap(old["selected_rows"], after),
                  "paired_total_effect": paired_bootstrap(old["baseline_rows"], after),
                  "selected_rows": after}
    _write_json_atomically(output / "audit.json", report)
    # Full fold metrics are descriptive: some images were used for selection.
    all_rows = score(ids, chosen)
    _write_json_atomically(output / "full_fold.json", {"summary": summarize(all_rows), "rows": all_rows})
    # Deterministic preview choices shared across old/new checkpoints.
    _, preview_ds = make_loader(coco, splits["monitor"], cfg, False)
    preview_cfg = replace(cfg, **chosen)
    preview_summary = summarize([r for r in all_rows if r["image_id"] in splits["monitor"]])
    history = [{"epoch": 0, "loss": float("nan"), "pq": preview_summary["mean_pq"],
                "rq": {"mean": preview_summary["rq"]}}]
    save_training_figures(model, preview_ds, preview_cfg, 0, history)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("rows")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "compare"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--parameters")
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.mode, args.baseline, args.parameters)
