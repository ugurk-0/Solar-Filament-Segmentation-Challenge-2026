"""Compare local and organizer-notebook counting on the fixed research split.

Source specification: kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
This keeps our deduplicated annotations and support mask; it is not a claim
to reproduce the hidden server's annotation selection or test distribution.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import pq_experiment as ex
import solution as sol


def main():
    root = Path("runs/pq_refinement_20260928_fixed")
    protocol = json.loads((root / "protocol.json").read_text())
    cache = Path("runs/pq_research_20260926/comparison_round5") / (
        "probabilities_" + protocol["checkpoint_sha256"][:12])
    cfg = replace(ex.config(root, fold=1), **json.loads((root / "best_postproc.json").read_text())["parameters"])
    coco, _, ids = sol.get_fold(cfg)
    _, ds = sol.make_loader(coco, ids, cfg, False)
    totals = dict(tp=0, fp=0, fn=0, matched_iou_sum=0.0)
    different = []
    for i in protocol["splits"]["audit"]:
        pred = sol.prob_to_instances(np.load(cache / f"{i}.npy"), cfg, ds.limb)
        gt = sol.ground_truth_instances(ds, i)
        iou, matches = sol.match_instances(pred, gt)
        hit = iou > 0.5
        tp = int(hit.sum())
        fp = int((~hit.any(axis=1)).sum())
        fn = int((~hit.any(axis=0)).sum())
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["matched_iou_sum"] += float(iou[hit].sum())
        if (tp, fp, fn) != (len(matches), len(pred) - len(matches), len(gt) - len(matches)):
            different.append(i)
    denominator = totals["tp"] + 0.5 * (totals["fp"] + totals["fn"])
    result = dict(organizer_counting_pq=totals["matched_iou_sum"] / denominator if denominator else 0,
                  totals=totals, observations_with_different_counts=different,
                  n_observations=len(protocol["splits"]["audit"]),
                  local_dataset_pq=json.loads((root / "assessment.json").read_text())["after"]["dataset_pq"],
                  scope="Same 85 deduplicated observations and support mask; not hidden-server verification")
    sol._write_json_atomically(root / "official_counting_audit.json", result)
    Path("reports/official_counting_audit_20260928.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
