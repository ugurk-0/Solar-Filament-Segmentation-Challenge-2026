"""Diagnose instance errors on the fixed assessment subset, never the test set."""
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

import pq_experiment as ex
import solution as sol


def main():
    root = Path("runs/pq_refinement_20260928_fixed")
    protocol = json.loads((root / "protocol.json").read_text())
    cfg = replace(ex.config(root), **json.loads((root / "best_postproc.json").read_text())["parameters"])
    cache = Path("runs/pq_research_20260926/comparison_round5") / (
        "probabilities_" + protocol["checkpoint_sha256"][:12])
    coco, _, ids = sol.get_fold(cfg)
    _, ds = sol.make_loader(coco, ids, cfg, False)
    gt_bins, missed_bins, missed_iou_bins = Counter(), Counter(), Counter()
    fp_area, fp_confidence, tp_confidence = [], [], []
    for i in protocol["splits"]["audit"]:
        probability = np.load(cache / f"{i}.npy")
        pred = sol.prob_to_instances(probability, cfg, ds.limb)
        gt = sol.ground_truth_instances(ds, i)
        iou, matches = sol.match_instances(pred, gt)
        matched_pred, matched_gt = {m[0] for m in matches}, {m[1] for m in matches}
        for j, mask in enumerate(gt):
            area = int(mask.sum())
            group = "under_500" if area < 500 else "500_to_1499" if area < 1500 else "1500_plus"
            gt_bins[group] += 1
            if j not in matched_gt:
                missed_bins[group] += 1
                best = float(iou[:, j].max()) if len(pred) else 0.0
                missed_iou_bins["under_0.1" if best < 0.1 else "0.1_to_0.3" if best < 0.3 else "0.3_to_0.5"] += 1
        for j, mask in enumerate(pred):
            confidence = float(probability[mask].mean())
            if j in matched_pred:
                tp_confidence.append(confidence)
            else:
                fp_confidence.append(confidence)
                fp_area.append(int(mask.sum()))
    stats = lambda values: {"n": len(values), "median": float(np.median(values)),
                            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))} if values else {"n": 0}
    result = dict(ground_truth_by_area=dict(gt_bins), missed_by_area=dict(missed_bins),
                  missed_best_iou=dict(missed_iou_bins), false_positive_area=stats(fp_area),
                  false_positive_mean_probability=stats(fp_confidence),
                  true_positive_mean_probability=stats(tp_confidence),
                  scope="85 reused research assessment observations; diagnostic, not a tuning objective")
    sol._write_json_atomically(root / "error_diagnosis.json", result)
    Path("reports/instance_errors_20260928.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
