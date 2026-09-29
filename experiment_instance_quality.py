"""Train, calibrate, and assess an instance-quality CNN with a frozen U-Net."""
import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import instance_quality as iq
import pq_experiment as ex
import solution as sol

ROOT = Path("runs/instance_quality_20260929")
PARENT = Path("runs/pq_research_20260926/training_round5/fold1_best.pt")
BASE = Path("runs/pq_refinement_20260928_fixed")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_group(folder, image_id, proposal):
    with np.load(folder / f"{image_id}_{proposal}.npz", allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def evaluate_grid(model, folder, ids, coco, device):
    rows = {(proposal, cutoff): [] for proposal in iq.PROPOSALS for cutoff in iq.CUTOFFS}
    for image_id in ids:
        for proposal in iq.PROPOSALS:
            group = load_group(folder, image_id, proposal)
            scores = iq.quality_scores(model, group["patches"], group["geometry"], device)
            for cutoff in iq.CUTOFFS:
                row = iq.matrix_metrics(group["iou"], scores >= cutoff)
                row.update(image_id=image_id, file_name=coco.imgs[image_id]["file_name"])
                rows[proposal, cutoff].append(row)
    grid = [{"proposal": p, "cutoff": c, "summary": iq.summarize(r)} for (p, c), r in rows.items()]
    return grid, rows


def export_candidate(root, state, cfg, quality_model):
    if not state.get("promoted"):
        raise ValueError("Only a promoted full-run candidate can be exported automatically")
    output = root / "submission"
    output.mkdir(exist_ok=True)
    if (output / "submission.csv").exists():
        raise FileExistsError("Submission already exists; preserve it")
    selected = state["selected"]
    cfg = replace(cfg, **iq.PROPOSALS[selected["proposal"]])
    model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
    payload = torch.load(PARENT, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["ema"])
    del payload
    paths = sorted(Path("MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError("No test JPEGs")
    valid, rows, image_counts = sol.build_limb_mask(cfg), [], {}
    for path in paths:
        image = sol.load_image(path)
        probability = sol.predict_full([model], image, cfg)
        masks = sol.prob_to_instances(probability, cfg, valid)
        masks = iq.filter_instances(quality_model, image, probability, masks, valid,
                                    selected["cutoff"], cfg.device)
        image_counts[path.name] = len(masks)
        occupied = np.zeros_like(valid)
        for index, mask in enumerate(masks, 1):
            if not mask.any() or np.any(mask & occupied) or np.any(mask & ~valid):
                raise ValueError("Invalid instance ownership/support during export")
            occupied |= mask
            counts = sol.mask_to_rle(mask)
            decoded = sol.maskutils.decode({"size": list(mask.shape), "counts": counts})
            if not np.array_equal(decoded.astype(bool), mask):
                raise ValueError("RLE round trip failed")
            rows.append({"filament_id": f"{path.stem}_{index}", "segmentation_rle": counts})
        print(f"export {path.name}: {len(masks)}", flush=True)
    with (output / "submission.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    sol._write_json_atomically(output / "submission_manifest.json", {
        "checkpoints": [str(PARENT), str(root / "quality_best.pt")],
        "checkpoint_sha256": digest(PARENT), "quality_sha256": digest(root / "quality_best.pt"),
        "config": vars(cfg), "quality_selection": selected,
        "image_instance_counts": image_counts, "n_images": len(paths), "n_instances": len(rows),
        "rle_round_trip": "all passed", "instance_overlap": "none"})


def main(smoke=False):
    root = ROOT / "smoke" if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    sol.seed_everything(2026)
    torch.set_num_threads(4)
    cfg = replace(ex.config(root), instance_method="components", close_kernel=1, thresh=0.8, min_area=500)
    coco, train_ids, val_ids = sol.get_fold(cfg)
    split = ex.partition(val_ids, cfg.seed)
    unique = {}
    for image_id in train_ids:
        unique.setdefault(sol.base_name(coco.imgs[image_id]["file_name"]), image_id)
    keys = sorted(unique)
    chosen_keys = np.random.default_rng(cfg.seed).choice(keys, min(6 if smoke else 128, len(keys)), replace=False)
    training = [unique[key] for key in chosen_keys]
    assert not set(chosen_keys) & {sol.base_name(coco.imgs[i]["file_name"]) for i in val_ids}
    if smoke:
        split = {name: ids[:2] for name, ids in split.items()}
    checkpoint_hash = digest(PARENT)
    old_protocol = json.loads((BASE / "protocol.json").read_text())
    if old_protocol["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Historical probability cache uses a different checkpoint")
    for field in ("tta", "window", "overlap", "img_size", "limb_geometry", "disk_radius", "limb_max_deg"):
        if old_protocol["config"][field] != getattr(cfg, field):
            raise ValueError("Historical inference cache mismatch: " + field)
    signature = {name: digest(name) for name in ("solution.py", "instance_quality.py", "experiment_instance_quality.py", cfg.train_json)}
    protocol = {"source_sha256": signature, "checkpoint_sha256": checkpoint_hash,
                "training_ids": training, "split": split, "config": vars(cfg),
                "proposals": iq.PROPOSALS, "cutoffs": iq.CUTOFFS, "patch_size": iq.PATCH_SIZE,
                "smoke": smoke, "epochs": 2 if smoke else 12,
                "selection": "checkpoint: monitor PQ; proposal/cutoff: calibration PQ; assessment: reused audit",
                "caveat": "Training proposals are in-sample for the frozen semantic model. No held-out labels train the quality model; repeated assessment is exploratory."}
    protocol = json.loads(json.dumps(protocol))
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Experiment inputs changed; use a new output directory")
    sol._write_json_atomically(protocol_path, protocol)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "preparing", "history": [], "smoke": smoke}
    if state["status"] == "complete":
        print("Completed experiment preserved.")
        return

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            Path("reports/instance_quality_20260929.md").write_text(
                "# Instance-quality CNN experiment\n\n"
                "Hypothesis: predicted instance quality separates false detections better than mean pixel probability. "
                "A permissive proposal set can also retain smaller objects. Frozen Round-5 U-Net, same fold/support/no-TTA. "
                "128 training observations; 16 monitor, 40 calibration, 85 reused assessment observations. "
                "No test labels or external pretrained weights are used.\n\n"
                "Inspired by [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html); "
                "this is a separate lightweight adaptation, not that architecture or its reported results.\n\n"
                "Run: `python experiment_instance_quality.py --smoke`, then `python experiment_instance_quality.py`.\n\n"
                "```json\n" + json.dumps(state, indent=2) + "\n```\n", encoding="utf-8")
    try:
        folder = root / "features"
        folder.mkdir(exist_ok=True)
        _, dataset = sol.make_loader(coco, training + val_ids, cfg, False)
        semantic = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
        payload = torch.load(PARENT, map_location="cpu", weights_only=True)
        semantic.load_state_dict(payload["ema"])
        del payload
        old_cache = Path("runs/pq_research_20260926/comparison_round5") / ("probabilities_" + checkpoint_hash[:12])
        # Assessment feature extraction is deferred until checkpoint/settings are frozen.
        prepare_ids = training + split["monitor"] + split["calibration"]

        def prepare(ids):
            for number, image_id in enumerate(ids, 1):
                if all((folder / f"{image_id}_{p}.npz").exists() for p in iq.PROPOSALS):
                    continue
                image, _, _ = dataset._load_full(image_id)
                cached = old_cache / f"{image_id}.npy"
                prob = np.load(cached) if cached.exists() else sol.predict_full([semantic], image, cfg)
                if prob.shape != image.shape or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                    raise ValueError("Invalid probability shape/range")
                gt = sol.ground_truth_instances(dataset, image_id)
                for proposal, params in iq.PROPOSALS.items():
                    masks = sol.prob_to_instances(prob, replace(cfg, **params), dataset.limb)
                    patches, geometry = iq.proposal_features(image, prob, masks, dataset.limb)
                    iou = sol.instance_iou_matrix(masks, gt).astype(np.float64)
                    best_iou = iou.max(axis=1) if len(gt) else np.zeros(len(masks), dtype=np.float64)
                    path = folder / f"{image_id}_{proposal}.npz"
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, patches=patches, geometry=geometry,
                                            iou=iou, best_iou=best_iou.astype(np.float32))
                    temporary.replace(path)
                state["prepared_images"] = number
                print(f"prepare {number}/{len(ids)} {image_id}", flush=True)
                if number % 10 == 0:
                    save()
        save()
        prepare(prepare_ids)
        semantic.to("cpu")
        torch.cuda.empty_cache()
        records = [load_group(folder, image_id, proposal) for image_id in training for proposal in iq.PROPOSALS]
        patches = np.concatenate([r["patches"] for r in records])
        geometry = np.concatenate([r["geometry"] for r in records])
        targets = np.concatenate([r["best_iou"] for r in records])
        del records
        if not len(targets) or not (targets >= 0.5).any() or not (targets < 0.5).any():
            raise ValueError("Quality training requires positive and negative proposals")
        state.update(status="training", training_proposals=len(targets),
                     positive_proposals=int((targets >= 0.5).sum()))
        save()
        trainset = torch.utils.data.TensorDataset(torch.from_numpy(patches), torch.from_numpy(geometry), torch.from_numpy(targets))
        loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        quality = iq.InstanceQualityNet().to(cfg.device)
        optimizer = torch.optim.AdamW(quality.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol["epochs"])
        start_epoch, best = 0, -1.0
        last_path = root / "quality_last.pt"
        if last_path.exists():
            saved = torch.load(last_path, map_location=cfg.device, weights_only=True)
            quality.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_epoch, best = saved["epoch"], saved["best_monitor_pq"]
            torch.set_rng_state(saved["rng"].cpu())
            if cfg.device == "cuda":
                torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
            del saved
        for epoch in range(start_epoch, protocol["epochs"]):
            quality.train()
            total, start = 0.0, time.monotonic()
            for patch, geom, target in loader:
                patch, geom, target = patch.to(cfg.device).float(), geom.to(cfg.device), target.to(cfg.device)
                patch = torch.rot90(patch, int(torch.randint(4, ()).item()), (-2, -1))
                if torch.rand(()).item() < 0.5:
                    patch = patch.flip(-1)
                prediction = quality(patch, geom)
                loss = F.binary_cross_entropy_with_logits(prediction[:, 0], (target >= 0.5).float())
                loss = loss + F.smooth_l1_loss(prediction[:, 1].sigmoid(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite quality loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(quality.parameters(), 5)
                optimizer.step()
                total += loss.item() * len(target)
            scheduler.step()
            grid, _ = evaluate_grid(quality, folder, split["monitor"], coco, cfg.device)
            monitor = max(grid, key=lambda row: row["summary"]["mean_pq"])
            pq = monitor["summary"]["mean_pq"]
            if pq > best:
                best = pq
                torch.save({"model": quality.state_dict(), "epoch": epoch + 1, "monitor": monitor,
                            "parent_sha256": checkpoint_hash, "smoke": smoke}, root / "quality_best.pt")
            state["history"] = [row for row in state["history"] if row["epoch"] < epoch + 1]
            state["history"].append(dict(epoch=epoch + 1, loss=total / len(targets), monitor_pq=pq,
                                         seconds=time.monotonic() - start))
            torch.save({"model": quality.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_monitor_pq": best,
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, last_path)
            save()
            print(state["history"][-1], flush=True)
        del loader, trainset, patches, geometry, targets, optimizer
        gc.collect()
        quality.load_state_dict(torch.load(root / "quality_best.pt", map_location=cfg.device, weights_only=True)["model"])
        state["status"] = "calibrating"
        save()
        grid, _ = evaluate_grid(quality, folder, split["calibration"], coco, cfg.device)
        selected = max(grid, key=lambda row: row["summary"]["mean_pq"])
        state.update(selected={key: selected[key] for key in ("proposal", "cutoff")}, calibration=grid, status="assessing")
        sol._write_json_atomically(root / "selection.json", {"selected": state["selected"], "calibration": grid})
        save()
        semantic.to(cfg.device)
        prepare(split["audit"])
        semantic.to("cpu")
        grid, rows = evaluate_grid(quality, folder, split["audit"], coco, cfg.device)
        before = rows["strict", 0.0]
        after = rows[selected["proposal"], selected["cutoff"]]
        baseline = iq.summarize(before)
        if not smoke:
            historical = json.loads((BASE / "assessment.json").read_text())["after"]
            if abs(baseline["dataset_pq"] - historical["dataset_pq"]) > 1e-9:
                raise ValueError("Baseline reproduction failed; comparison invalid")
        paired = ex.paired_bootstrap(before, after)
        state["assessment"] = {"baseline": baseline, "candidate": iq.summarize(after), "paired": paired}
        state["promoted"] = bool(not smoke and state["assessment"]["candidate"]["dataset_pq"] > baseline["dataset_pq"]
                                 and paired["observation_bootstrap_95ci"][0] > 0)
        sol._write_json_atomically(root / "assessment.json", {**state["assessment"], "baseline_rows": before, "selected_rows": after})
        if state["promoted"]:
            state["status"] = "exporting_candidate"
            save()
            del semantic
            gc.collect()
            torch.cuda.empty_cache()
            export_candidate(root, state, cfg, quality)
        state.update(status="complete", leaderboard_score=None)
        save()
        print(json.dumps(state["assessment"], indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
