"""Controlled inference-context experiment; no test images or labels are used."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

import pq_experiment as ex
import solution as sol

ROOT = Path("runs/window_context_20260929")
PARENT = Path("runs/pq_research_20260926/training_round5/fold1_best.pt")
BASE = Path("runs/pq_refinement_20260928_fixed")
REPORT = Path("reports/window_context_20260929.md")
WINDOWS = [(1024, 256), (768, 256), (512, 128)]


def save(state):
    sol._write_json_atomically(ROOT / "state.json", state)
    text = ["# Inference context experiment — 29 September 2026", "",
            "Target: PQ 0.60; not an achieved score.", "",
            "Hypothesis: matching the training crop context can improve inference with GroupNorm. "
            "The checkpoint, physical-observation split, support mask and no-TTA setting are fixed. "
            "Window/overlap pairs are 1024/256, 768/256, and 512/128. Thresholds are "
            "0.65/0.80/0.90; minimum areas are 200/500. Selection uses mean PQ on the "
            "40 calibration observations. The frozen winner is compared on 85 reused research "
            "observations; this is exploratory, not an untouched or leaderboard estimate.", "",
            "Command: `python experiment_window_context.py`", "",
            f"Status: **{state['status']}**", "",
            "| Window | Overlap | Threshold | Minimum area | Calibration mean PQ | Calibration pooled PQ |",
            "|---:|---:|---:|---:|---:|---:|"]
    for row in state.get("grid", []):
        p, s = row["parameters"], row["summary"]
        text.append(f"| {p['window']} | {p['overlap']} | {p['thresh']} | {p['min_area']} | {s['mean_pq']:.6f} | {s['dataset_pq']:.6f} |")
    if "assessment" in state:
        text += ["", "Frozen-setting assessment:", "", "```json",
                 json.dumps(state["assessment"], indent=2), "```"]
    if "error" in state:
        text += ["", "Error: " + state["error"]]
    REPORT.write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = replace(ex.config(ROOT), instance_method="components", close_kernel=1,
                  thresh=0.8, min_area=500)
    if cfg.device != "cuda":
        raise RuntimeError("This bounded full-resolution experiment requires the available CUDA GPU")
    checkpoint_hash = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    original = json.loads((BASE / "protocol.json").read_text())
    if checkpoint_hash != original["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint hash mismatch")
    for name in ("window", "overlap", "tta", "limb_geometry", "img_size", "disk_radius", "limb_max_deg"):
        if original["config"][name] != getattr(cfg, name):
            raise ValueError("Baseline cache configuration mismatch: " + name)
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in ("solution.py", "pq_experiment.py", __file__)}
    state_path = ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "calibrating", "checkpoint_sha256": checkpoint_hash,
        "source_sha256": source_hashes, "grid": []}
    if state["source_sha256"] != source_hashes or state["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Source/checkpoint changed; preserve this experiment and use a new run directory")
    if state["status"] == "complete":
        print("Experiment already complete; results preserved.")
        return
    coco, _, ids = sol.get_fold(cfg)
    splits = ex.partition(ids, cfg.seed)
    if splits != original["splits"]:
        raise ValueError("Observation partition differs from the recorded baseline")
    sol._write_json_atomically(ROOT / "protocol.json", {
        "config": vars(cfg), "splits": splits, "checkpoint_sha256": checkpoint_hash,
        "windows": WINDOWS, "thresholds": [0.65, 0.8, 0.9], "minimum_areas": [200, 500],
        "selection_metric": "calibration mean image PQ", "source_sha256": source_hashes})
    _, ds = sol.make_loader(coco, ids, cfg, False)
    model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
    payload = torch.load(PARENT, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["ema"])
    del payload
    old_cache = Path("runs/pq_research_20260926/comparison_round5") / ("probabilities_" + checkpoint_hash[:12])

    def probability(image_id, window, overlap):
        folder = old_cache if window == 1024 and overlap == 256 else ROOT / f"probabilities_{window}_{overlap}"
        path = folder / f"{image_id}.npy"
        if path.exists():
            prob = np.load(path)
        else:
            image, _, _ = ds._load_full(image_id)
            prob = sol.predict_full([model], image, replace(cfg, window=window, overlap=overlap))
            if folder == old_cache:
                # Never modify the historical cache if an entry is absent.
                folder = ROOT / f"probabilities_{window}_{overlap}"
                path = folder / f"{image_id}.npy"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(path, prob)
        if prob.shape != (cfg.img_size, cfg.img_size) or not np.isfinite(prob).all():
            raise ValueError(f"Invalid probability map for {image_id}")
        if prob.min() < 0 or prob.max() > 1:
            raise ValueError("Probability outside [0,1]")
        return prob

    try:
        save(state)
        for window, overlap in WINDOWS:
            settings = [dict(window=window, overlap=overlap, thresh=t, min_area=a)
                        for t in (0.65, 0.8, 0.9) for a in (200, 500)]
            if all(any(r["parameters"] == p for r in state["grid"]) for p in settings):
                continue
            rows = [[] for _ in settings]
            for number, image_id in enumerate(splits["calibration"], 1):
                start = time.monotonic()
                prob = probability(image_id, window, overlap)
                gt = sol.ground_truth_instances(ds, image_id)
                for index, setting in enumerate(settings):
                    pred = sol.prob_to_instances(prob, replace(cfg, **setting), ds.limb)
                    row = sol.image_metrics(pred, gt)
                    row.update(image_id=image_id, file_name=coco.imgs[image_id]["file_name"])
                    rows[index].append(row)
                print(f"calibration window={window} image={number}/40 seconds={time.monotonic()-start:.1f}", flush=True)
            state["grid"] = [r for r in state["grid"] if r["parameters"]["window"] != window]
            state["grid"] += [{"parameters": p, "summary": ex.summarize(r)} for p, r in zip(settings, rows)]
            save(state)
        winner = max(state["grid"], key=lambda r: r["summary"]["mean_pq"])
        state.update(status="assessing_frozen_winner", selected=winner["parameters"])
        save(state)
        sol._write_json_atomically(ROOT / "selected_parameters.json", {
            "parameters": dict(instance_method="components", close_kernel=1, **winner["parameters"]),
            "calibration": winner["summary"]})
        selected = winner["parameters"]
        rows = []
        for number, image_id in enumerate(splits["audit"], 1):
            prob = probability(image_id, selected["window"], selected["overlap"])
            gt = sol.ground_truth_instances(ds, image_id)
            pred = sol.prob_to_instances(prob, replace(cfg, **selected), ds.limb)
            row = sol.image_metrics(pred, gt)
            row.update(image_id=image_id, file_name=coco.imgs[image_id]["file_name"])
            rows.append(row)
            print(f"assessment image={number}/85", flush=True)
        baseline = json.loads((BASE / "assessment.json").read_text())
        comparison = ex.paired_bootstrap(baseline["selected_rows"], rows)
        state["assessment"] = {"baseline": baseline["after"], "selected": ex.summarize(rows),
                               "paired_vs_baseline": comparison,
                               "target_pq": 0.6, "leaderboard_score": None}
        sol._write_json_atomically(ROOT / "assessment.json", {**state["assessment"], "selected_rows": rows})
        state["status"] = "complete"
        save(state)
        print(json.dumps(state["assessment"], indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(state)
        raise


if __name__ == "__main__":
    main()
