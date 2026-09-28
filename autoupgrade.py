"""Run a resumable, reported batch of controlled PQ improvement experiments.

Each candidate changes one training setting from Round 5. Monitoring screens
candidates before the reused assessment split is inspected. No test labels are
read. Completed runs are never restarted or overwritten automatically.
"""
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

import pq_experiment as ex
import solution as sol

ROOT = Path("runs/autoupgrade_20260928")
PARENT = Path("runs/pq_research_20260926/training_round5/fold1_best.pt")
CALIBRATION = Path("runs/pq_refinement_20260928_fixed")
REPORT = Path("reports/autoupgrade_20260928.md")
EXPERIMENTS = [
    ("less_cldice", "Reducing skeleton-loss weight may recover complete masks and improve instance IoU.",
     dict(w_cld=0.2, w_dice=0.5)),
    ("more_random_crops", "Reducing filament-centered sampling to 25% may suppress remaining false detections.",
     dict(oversample_p=0.25)),
]


def write_state(state):
    sol._write_json_atomically(ROOT / "state.json", state)
    lines = ["# Automatic improvement batch — 28 September 2026", "",
             "User-reported Kaggle baseline: **0.24**. No automatic leaderboard claim.", "",
             "Each run starts from Round 5, using the fixed observation split and corrected post-processing. "
             "Epochs are selected on 16 monitor observations. Only candidates exceeding the starting monitor "
             "PQ by 0.005 are assessed on the reused 85-observation research split. Promotion requires gains "
             "in both mean and pooled PQ and a positive lower paired-bootstrap bound. This conservative gate "
             "does not remove selection bias from repeated research. Test labels are never read.", "",
             "Progress and epoch previews: `runs/autoupgrade_20260928/<experiment>/`. "
             "Reproduce with `python autoupgrade.py`. Completed runs are preserved.", "",
             "## Export defect found before this batch", "",
             "The previous candidate export stopped on overlapping instances at 20171021223430Lh.jpeg. "
             "Small-hole filling could absorb a nested component. Cleanup now falls back to the original "
             "component whenever filled pixels would collide with another instance. Two synthetic regression "
             "tests cover nested ownership and ordinary hole filling. Calibration and assessment are rerun "
             "with the corrected converter before this batch.", "",
             f"Starting monitor PQ: {state.get('baseline_monitor_pq', 'pending')}", ""]
    lines.extend([f"Corrected candidate export: {state.get('export_status', 'pending')}", ""])
    for name, record in state.get("experiments", {}).items():
        lines.extend([f"## {name}", "", record["hypothesis"], "",
                      f"Status: **{record['status']}**", "",
                      "```json", json.dumps(record, indent=2), "```", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state_path = ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"experiments": {}}
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in ("solution.py", "autoupgrade.py", "pq_experiment.py")}
    if "source_sha256" in state and state["source_sha256"] != source_hashes:
        raise RuntimeError("Source changed since this batch started; use a new batch directory")
    state["source_sha256"] = source_hashes
    settings = json.loads((CALIBRATION / "best_postproc.json").read_text())["parameters"]
    base = ex.config(ROOT, fold=1)
    base = replace(base, **settings, epochs=3, lr=1e-5, ema_decay=0.99,
                   eval_every=1, eval_max_images=16, init_ckpt=str(PARENT),
                   pos_weight_cap=1.0, oversample_p=0.5)
    if not (CALIBRATION / "submission_manifest.json").exists():
        state["export_status"] = "running"
        write_state(state)
        try:
            sol.submit(replace(base, work_dir=str(CALIBRATION),
                               test_dir="MAGFiLO_1.0_Kaggle_2026/test/test_images"), [str(PARENT)])
        except Exception as error:
            state["export_status"] = f"failed: {type(error).__name__}: {error}"
            write_state(state)
            raise
    state["export_status"] = "complete"
    if "baseline_monitor_pq" not in state:
        coco, _, ids = sol.get_fold(base)
        _, ds = sol.make_loader(coco, ex.partition(ids, base.seed)["monitor"], base, False)
        model = sol.FilamentUNet(base, pretrained=False).to(base.device)
        payload = torch.load(PARENT, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["ema"])
        report = sol.evaluate_detailed(model, ds, base)
        state["baseline_monitor_pq"] = report["aggregate"]["pq"]["mean"]
        sol._write_json_atomically(ROOT / "baseline_monitor.json", report)
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
    write_state(state)
    baseline = json.loads((CALIBRATION / "assessment.json").read_text())
    for name, hypothesis, changes in EXPERIMENTS:
        record = state["experiments"].get(name)
        if record and record["status"] in ("rejected_monitor", "rejected_assessment", "candidate_uncertain", "promoted"):
            continue
        cfg = replace(base, work_dir=str(ROOT / name), **changes)
        record = dict(hypothesis=hypothesis, status="training", config=vars(cfg))
        state["experiments"][name] = record
        write_state(state)
        def epoch_finished(epoch, checkpoint):
            record["completed_epochs"] = epoch
            record["history"] = [{k: r[k] for k in ("epoch", "loss", "pq", "lr")}
                                 for r in json.loads((Path(cfg.work_dir) / "history.json").read_text())]
            write_state(state)
        try:
            # Completed training can be assessed after a restart; interrupted
            # training is deliberately not overwritten by sol.train's guard.
            history_path = Path(cfg.work_dir) / "history.json"
            if not history_path.exists() or len(json.loads(history_path.read_text())) < cfg.epochs:
                sol.train(cfg, epoch_callback=epoch_finished)
            history = json.loads(history_path.read_text())
            best = max(r["pq"] for r in history)
            record.update(best_monitor_pq=best,
                          history=[{k: r[k] for k in ("epoch", "loss", "pq", "lr")} for r in history])
            if best < state["baseline_monitor_pq"] + 0.005:
                record["status"] = "rejected_monitor"
                write_state(state)
                continue
            record["status"] = "assessing"
            write_state(state)
            output = ROOT / (name + "_assessment")
            ex.run(Path(cfg.work_dir) / "fold1_best.pt", output, "compare",
                   "runs/pq_research_20260926/baseline/audit.json",
                   CALIBRATION / "best_postproc.json")
            measured = json.loads((output / "audit.json").read_text())
            paired = ex.paired_bootstrap(baseline["selected_rows"], measured["selected_rows"])
            record.update(assessment=measured["selected"], paired_vs_parent=paired)
            improved = (paired["mean_pq_delta"] >= 0.005 and
                        measured["selected"]["dataset_pq"] > baseline["after"]["dataset_pq"])
            record["status"] = ("promoted" if improved and paired["observation_bootstrap_95ci"][0] > 0
                                else "candidate_uncertain" if improved else "rejected_assessment")
            write_state(state)
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_state(state)
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    state["batch_status"] = "complete"
    write_state(state)


if __name__ == "__main__":
    main()
