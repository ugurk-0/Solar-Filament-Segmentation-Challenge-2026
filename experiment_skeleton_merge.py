"""Calibration-gated test of skeleton-endpoint merging on cached probabilities."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import pq_experiment as ex
import solution as sol
from skeleton_merge import merge_instances

ROOT = Path("runs/skeleton_merge_20261003")
REPORT = Path("reports/skeleton_merge_20261003.md")
BASE = Path("runs/pq_refinement_20260928_fixed")
GRID = [
    {"max_distance": d, "min_cosine": c, "min_gap_probability": p}
    for d in (25, 50)
    for c in (0.7, 0.8)
    for p in (0.3, 0.4)
]


def score(ids, cache, cfg, limb, ds, parameters=None):
    rows = []
    for index, image_id in enumerate(ids, 1):
        probability = np.load(cache / f"{image_id}.npy")
        pred = sol.prob_to_instances(probability, cfg, limb)
        if parameters is not None:
            pred = merge_instances(pred, probability, **parameters)
        row = sol.image_metrics(pred, sol.ground_truth_instances(ds, image_id))
        row.update(image_id=image_id, file_name=ds.coco.imgs[image_id]["file_name"])
        rows.append(row)
        print(f"{index}/{len(ids)} id={image_id} PQ={row['pq']:.4f} "
              f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}", flush=True)
    return rows


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((BASE / "protocol.json").read_text())
    cfg = replace(ex.config(BASE),
                  **json.loads((BASE / "best_postproc.json").read_text())["parameters"])
    cache = Path("runs/pq_research_20260926/comparison_round5") / (
        "probabilities_" + protocol["checkpoint_sha256"][:12])
    coco, _, ids = sol.get_fold(cfg)
    split = ex.partition(ids, cfg.seed)
    _, ds = sol.make_loader(coco, ids, cfg, False)
    state_path = ROOT / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("status") == "complete":
            print("Completed experiment preserved.")
            return
    else:
        state = {"status": "calibrating", "grid": GRID}
    grid_path = ROOT / "grid.json"
    grid = json.loads(grid_path.read_text()) if grid_path.exists() else []
    for setting in GRID:
        if any(row["parameters"] == setting for row in grid):
            continue
        rows = score(split["calibration"], cache, cfg, ds.limb, ds, setting)
        grid.append({"parameters": setting, "summary": ex.summarize(rows)})
        sol._write_json_atomically(grid_path, grid)
    selected = max(grid, key=lambda row: row["summary"]["mean_pq"])
    state.update(status="assessing", selection=selected)
    sol._write_json_atomically(state_path, state)
    before = score(split["audit"], cache, cfg, ds.limb, None)
    after = score(split["audit"], cache, cfg, ds.limb, selected["parameters"])
    assessment = {"baseline": ex.summarize(before), "merged": ex.summarize(after),
                  "paired": ex.paired_bootstrap(before, after),
                  "parameters": selected["parameters"]}
    sol._write_json_atomically(ROOT / "assessment.json",
                               {**assessment, "baseline_rows": before, "merged_rows": after})
    promoted = bool(assessment["merged"]["dataset_pq"] > assessment["baseline"]["dataset_pq"]
                    and assessment["paired"]["observation_bootstrap_95ci"][0] > 0)
    state.update(status="complete", assessment=assessment, promoted=promoted,
                 decision="promoted_research_candidate" if promoted else "rejected_assessment")
    sol._write_json_atomically(state_path, state)
    REPORT.write_text(
        "# Skeleton-endpoint graph merge (Qiuwei V2 reproduction)\n\n"
        "Post-processing only: no retraining, no test data. Merge parameters tuned on the "
        "40 calibration observations; comparison on the 85 reused audit observations against "
        "the identical cached probabilities with components-only decoding.\n\n"
        "```json\n" + json.dumps(state, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in state.items() if k != "grid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
