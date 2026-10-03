"""Matched raw-versus-CLAHE fine-tuning reproduction; no test-set tuning."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import pq_experiment as ex
import solution as sol
from clahe_preprocessing import clahe_equalize
from instance_quality import summarize

ROOT = Path("runs/clahe_20261002")
PARENT = Path("runs/pq_research_20260926/training_round5/fold1_best.pt")
BASE = Path("runs/pq_refinement_20260928_fixed")
REPORT = Path("reports/clahe_20261002.md")
OriginalDataset = sol.FilamentDataset


class ClaheDataset(OriginalDataset):
    def _load_full(self, image_id):
        if image_id in self._cache:
            return self._cache[image_id]
        image, mask, auxiliary = super()._load_full(image_id)
        if getattr(self.cfg, "clahe_enabled", False):
            image = clahe_equalize(
                image,
                kernel_size=getattr(self.cfg, "clahe_kernel_size", None),
                clip_limit=getattr(self.cfg, "clahe_clip_limit", 0.01),
                nbins=getattr(self.cfg, "clahe_nbins", 256),
            )
        result = image, mask, auxiliary
        self._cache[image_id] = result
        return result


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(state, smoke):
    REPORT.write_text(
        "# CLAHE matched fine-tuning reproduction\n\n"
        "Hypothesis: a controlled CLAHE reproduction improves local contrast enough to "
        "increase monitor PQ over both the retained Round-5 checkpoint and a matched raw "
        "fine-tuning control. This experiment reuses the same physical-observation split, "
        "support mask, checkpoint initialization, and post-processing protocol. Selection "
        "remains restricted to 16 monitor observations, calibration remains 40 observations, "
        "and assessment remains 85 reused research observations. No test images or labels "
        "are used.\n\n"
        f"Current status was recorded from {'`runs/clahe_20261002/smoke`' if smoke else '`runs/clahe_20261002`'}.\n\n"
        "Reproduce: `python experiment_clahe.py --smoke`, then `python experiment_clahe.py`. "
        "CLAHE checkpoints require ClaheDataset or explicit `clahe_equalize` before inference; "
        "the ordinary CLI does not apply that transform.\n\n"
        "```json\n" + json.dumps(state, indent=2) + "\n```\n",
        encoding="utf-8",
    )


def main(smoke=False):
    torch.set_num_threads(4)
    root = ROOT / "smoke" if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        ex.config(root),
        thresh=0.8,
        min_area=500,
        close_kernel=1,
        epochs=2 if smoke else 3,
        lr=1e-5,
        ema_decay=0.99,
        eval_every=1,
        eval_max_images=16,
        init_ckpt=str(PARENT),
        pos_weight_cap=1,
        oversample_p=0.5,
        preview_count=0,
        limit=4 if smoke else 0,
    )
    original_fold = sol.get_fold
    coco, train, validation = original_fold(replace(cfg, limit=0))
    groups = {}
    for image_id in train:
        groups.setdefault(sol.base_name(coco.imgs[image_id]["file_name"]), image_id)
    keys = sorted(groups)
    chosen = np.random.default_rng(cfg.seed).choice(keys, min(4 if smoke else 128, len(keys)), replace=False)
    selected = [groups[key] for key in chosen]
    assert not set(chosen) & {sol.base_name(coco.imgs[i]["file_name"]) for i in validation}
    splits = ex.partition(validation, cfg.seed)
    protocol = dict(
        config=vars(cfg),
        training_ids=selected,
        split=splits,
        arms={
            "raw_control": {"clahe_enabled": False},
            "clahe_qiuwei": {"clahe_enabled": True, "clahe_clip_limit": 0.01, "clahe_nbins": 256},
        },
        smoke=smoke,
        checkpoint_sha256=digest(PARENT),
        source_sha256={
            name: digest(name)
            for name in (
                "solution.py",
                "clahe_preprocessing.py",
                "experiment_clahe.py",
                cfg.train_json,
            )
        },
        selection=(
            "Three epochs per arm in the full run, two in smoke; checkpoint by 16 monitor "
            "observations; assess CLAHE only if it beats the parent and raw control by 0.005"
        ),
    )
    protocol = json.loads(json.dumps(protocol))
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Inputs changed; use a new output directory")
    sol._write_json_atomically(path, protocol)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="training", arms={})
    if state["status"] == "complete":
        print("Completed pilot preserved.")
        return

    def save():
        sol._write_json_atomically(state_path, state)
        write_report(state, smoke)

    def fixed_fold(current_cfg):
        if current_cfg.fold != cfg.fold:
            raise ValueError("Unexpected fold")
        return coco, selected, validation

    try:
        sol.get_fold, sol.FilamentDataset = fixed_fold, ClaheDataset
        for arm, settings in protocol["arms"].items():
            arm_cfg = replace(cfg, work_dir=str(root / arm))
            for key, value in settings.items():
                setattr(arm_cfg, key, value)
            history_path = Path(arm_cfg.work_dir) / "history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            state.update(status="training", active_arm=arm)
            save()
            if len(history) < cfg.epochs:
                def finished(epoch, checkpoint):
                    state["arms"][arm] = json.loads(history_path.read_text())
                    save()

                sol.train(arm_cfg, epoch_callback=finished)
            history = json.loads(history_path.read_text())
            state["arms"][arm] = [
                {key: row[key] for key in ("epoch", "loss", "pq", "elapsed_seconds")}
                for row in history
            ]
            gc.collect()
            torch.cuda.empty_cache()
            save()
        parent_monitor = json.loads(Path("runs/autoupgrade_20260928/baseline_monitor.json").read_text())["aggregate"]["pq"]["mean"]
        raw = max(row["pq"] for row in state["arms"]["raw_control"])
        clahe = max(row["pq"] for row in state["arms"]["clahe_qiuwei"])
        state.update(
            parent_monitor_pq=parent_monitor,
            raw_monitor_pq=raw,
            clahe_monitor_pq=clahe,
            decision="smoke_only" if smoke else "rejected_monitor",
            promoted=False,
        )
        if not smoke and clahe > max(raw, parent_monitor) + 0.005:
            state.update(status="assessing", decision="assessment_pending")
            save()
            assess_cfg = replace(cfg, work_dir=str(root / "clahe_qiuwei"))
            assess_cfg.clahe_enabled = True
            assess_cfg.clahe_clip_limit = 0.01
            assess_cfg.clahe_nbins = 256
            dataset = ClaheDataset(coco, splits["audit"], assess_cfg, False, sol.build_limb_mask(assess_cfg))
            model = sol.FilamentUNet(assess_cfg, pretrained=False).to(cfg.device)
            payload = torch.load(root / "clahe_qiuwei/fold1_best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(payload["ema"])
            del payload
            report = sol.evaluate_detailed(model, dataset, assess_cfg)
            baseline = json.loads((BASE / "assessment.json").read_text())
            paired = ex.paired_bootstrap(baseline["selected_rows"], report["per_image"])
            summary = summarize(report["per_image"])
            sol._write_json_atomically(root / "assessment.json", dict(report=report, paired=paired))
            state["assessment"] = dict(summary=summary, paired=paired)
            state["promoted"] = bool(
                summary["dataset_pq"] > baseline["after"]["dataset_pq"]
                and paired["observation_bootstrap_95ci"][0] > 0
            )
            state["decision"] = "promoted_research_candidate" if state["promoted"] else "rejected_assessment"
        state.update(status="complete", leaderboard_score=None)
        save()
        print(json.dumps(state, indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        sol.get_fold, sol.FilamentDataset = original_fold, OriginalDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)