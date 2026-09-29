"""Test one actual, representative annotation set per training observation.

No synthetic targets are created. Within each training-only duplicate group,
select the annotation set with the highest mean pairwise instance PQ. Ties use
image ID ordering. Validation IDs and annotation choice remain unchanged.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import pq_experiment as ex
import solution as sol

ROOT = Path("runs/annotation_medoid_20260929")
PARENT = Path("runs/pq_research_20260926/training_round5/fold1_best.pt")
BASE = Path("runs/pq_refinement_20260928_fixed")


def choose_records(coco, train_ids, limb):
    groups = defaultdict(list)
    for image_id in train_ids:
        groups[sol.base_name(coco.imgs[image_id]["file_name"])].append(image_id)
    selected, choices = [], []
    for name, members in sorted(groups.items()):
        members = sorted(members, key=str)
        scores = np.zeros(len(members))
        if len(members) > 1:
            masks = []
            for image_id in members:
                info = coco.imgs[image_id]
                gt = [sol.polygon_to_mask(ann, info["height"], info["width"]) & limb
                      for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))]
                masks.append([mask for mask in gt if mask.any()])
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pq = sol.panoptic_quality(masks[i], masks[j])[0]
                    scores[i] += pq
                    scores[j] += pq
            scores /= len(members) - 1
        chosen = members[int(scores.argmax())]
        selected.append(chosen)
        choices.append(dict(observation=name, candidates=members,
                            mean_pair_pq=scores.tolist(), chosen=chosen))
    return selected, choices


def main(smoke=False):
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    params = json.loads((BASE / "best_postproc.json").read_text())["parameters"]
    cfg = replace(ex.config(ROOT / ("smoke" if smoke else "training")), **params,
                  epochs=2 if smoke else 6, limit=8 if smoke else 0,
                  lr=1e-5, ema_decay=0.99, eval_every=1,
                  eval_max_images=8 if smoke else 16, init_ckpt=str(PARENT),
                  pos_weight_cap=1.0, oversample_p=0.5)
    original_get_fold = sol.get_fold
    coco, train_ids, validation_ids = original_get_fold(replace(cfg, limit=0))
    signature = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                 for name in ("solution.py", "pq_experiment.py", "experiment_annotation_medoid.py", cfg.train_json)}
    signature["checkpoint"] = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    protocol_file = ROOT / "protocol.json"
    if protocol_file.exists():
        protocol = json.loads(protocol_file.read_text())
        if protocol["sha256"] != signature:
            raise ValueError("Inputs changed; use a new experiment directory")
        selected = protocol["selected_train_ids"]
    else:
        selected, choices = choose_records(coco, train_ids, sol.build_limb_mask(cfg))
        train_keys = {sol.base_name(coco.imgs[i]["file_name"]) for i in selected}
        val_keys = {sol.base_name(coco.imgs[i]["file_name"]) for i in validation_ids}
        assert not train_keys & val_keys
        assert len(train_keys) == len(selected) and set(selected) <= set(train_ids)
        protocol = dict(sha256=signature, selected_train_ids=selected,
                        validation_ids=validation_ids, original_train_records=len(train_ids),
                        unique_train_observations=len(selected), choices=choices,
                        hypothesis="One actual representative annotation per observation reduces contradictory supervision and duplicate weighting",
                        limitations="Two-annotator groups tie; deterministic selection cannot identify the correct annotator. This combines target selection and observation reweighting.")
        sol._write_json_atomically(protocol_file, protocol)

    def selected_fold(current_cfg):
        current_coco, _, current_val = original_get_fold(replace(current_cfg, limit=0))
        if current_cfg.fold != cfg.fold or current_val != validation_ids:
            raise ValueError("Split changed during experiment")
        return (current_coco, selected[:current_cfg.limit] if current_cfg.limit else selected,
                current_val[:min(current_cfg.limit, len(current_val))] if current_cfg.limit else current_val)

    state_path = ROOT / ("smoke_state.json" if smoke else "state.json")
    state = dict(status="training", target_pq=0.6, configuration=vars(cfg),
                 selected_training_observations=len(selected), leaderboard_score=None)
    history_path = Path(cfg.work_dir) / "history.json"
    if state_path.exists() and json.loads(state_path.read_text())["status"] == "complete":
        print("Experiment already complete; results preserved.")
        return

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            Path("reports/annotation_medoid_20260929.md").write_text(
                "# Annotation-medoid experiment — 29 September 2026\n\n"
                + protocol["hypothesis"] + ".\n\n" + protocol["limitations"] + "\n\n"
                + "Fixed parent, fold, no TTA, support mask, and post-processing. Six epochs; checkpoint selected on 16 monitor observations. "
                + "The 85-observation assessment is reused research data. No test labels are used.\n\n"
                + "Reproduce: `python experiment_annotation_medoid.py --smoke`, then `python experiment_annotation_medoid.py`.\n\n"
                + "```json\n" + json.dumps(state, indent=2) + "\n```\n", encoding="utf-8")
    save()
    try:
        sol.get_fold = selected_fold
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        if len(history) < cfg.epochs:
            def finished(epoch, checkpoint):
                state["completed_epochs"] = epoch
                state["history"] = [{key: row[key] for key in ("epoch", "loss", "pq", "lr")}
                                    for row in json.loads(history_path.read_text())]
                save()
            sol.train(cfg, epoch_callback=finished)
        if smoke:
            state["status"] = "evaluating_smoke"
            save()
            sol.evaluate_folds(cfg)
        else:
            state["status"] = "assessing"
            save()
            # ex.run uses the unchanged historical split and inference config.
            sol.get_fold = original_get_fold
            out = ROOT / "assessment"
            ex.run(Path(cfg.work_dir) / "fold1_best.pt", out, "compare",
                   "runs/pq_research_20260926/baseline/audit.json", BASE / "best_postproc.json")
            report = json.loads((out / "audit.json").read_text())
            baseline = json.loads((BASE / "assessment.json").read_text())
            paired = ex.paired_bootstrap(baseline["selected_rows"], report["selected_rows"])
            state["assessment"] = {"selected": report["selected"], "baseline": baseline["after"], "paired": paired}
            state["promoted"] = bool(report["selected"]["dataset_pq"] > baseline["after"]["dataset_pq"]
                                     and paired["observation_bootstrap_95ci"][0] > 0)
        state["status"] = "complete"
        save()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        sol.get_fold = original_get_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
