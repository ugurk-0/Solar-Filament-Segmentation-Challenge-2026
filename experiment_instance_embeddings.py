"""Controlled experiment for instance-label embeddings on filament segmentation."""
import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

import instance_embeddings as ie
import pq_experiment as ex
import solution as sol


ROOT = Path("runs/instance_embeddings_20261002")
REPORT = Path("reports/instance_embeddings_20261002.md")
PARENT = Path("runs/pq_research_20260926/training_round5/fold1_best.pt")
CONTROL_BASE = Path("runs/pq_refinement_20260928_fixed/best_postproc.json")
MONITOR_MIN_DELTA = 0.0
EMBED_DEFAULT = {"threshold": 0.8, "min_area": 500, "eps": 0.7, "min_samples": 8}
EMBED_GRID = [
    {"threshold": threshold, "min_area": min_area, "eps": eps, "min_samples": min_samples}
    for threshold in (0.65, 0.8)
    for min_area in (200, 500)
    for eps in (0.5, 0.7)
    for min_samples in (4, 8)
]
ARM_SETTINGS = {
    "control": {"w_embed": 0.0, "embedding_dim": 0, "base_lr": 1e-5, "embed_lr": 0.0},
    "embedding": {"w_embed": 0.1, "embedding_dim": 8, "base_lr": 1e-5, "embed_lr": 1e-3},
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def control_parameters():
    if CONTROL_BASE.exists():
        return json.loads(CONTROL_BASE.read_text(encoding="utf-8"))["parameters"]
    return {"instance_method": "components", "thresh": 0.8, "min_area": 500, "close_kernel": 1}


def default_cfg(work_dir, smoke):
    cfg = replace(
        ex.config(work_dir),
        epochs=1 if smoke else 3,
        limit=8 if smoke else 0,
        lr=1e-5,
        ema_decay=0.99,
        oversample_p=0.5,
        pos_weight_cap=1.0,
        preview_count=0,
        eval_every=1,
        tta=False,
    )
    for key, value in control_parameters().items():
        setattr(cfg, key, value)
    return cfg


def build_instance_labels(coco, img_id, height, width):
    labels = np.zeros((height, width), np.int64)
    for instance_id, ann in enumerate(coco.loadAnns(coco.getAnnIds(imgIds=[img_id])), start=1):
        labels[sol.polygon_to_mask(ann, height, width)] = instance_id
    return labels


class InstanceLabelDataset(sol.FilamentDataset):
    """FilamentDataset with aligned per-instance integer labels."""

    def _load_full(self, img_id):
        cached = self._cache.get(img_id)
        if cached is not None and len(cached) == 4:
            return cached
        image, mask, auxiliary = super()._load_full(img_id)
        labels = build_instance_labels(self.coco, img_id, mask.shape[0], mask.shape[1])
        return image, mask, auxiliary, labels

    def __getitem__(self, i):
        image, mask, auxiliary, labels = self._load_full(self.ids[i])
        height, width = image.shape
        crop = self.cfg.crop_size
        if crop > min(height, width):
            raise ValueError("crop_size exceeds image dimensions")
        if self.train and random.random() < self.cfg.oversample_p and (mask & self.limb).any():
            ys, xs = np.where(mask & self.limb)
            point = random.randrange(len(ys))
            y0 = np.clip(int(ys[point]) - crop // 2, 0, height - crop)
            x0 = np.clip(int(xs[point]) - crop // 2, 0, width - crop)
        else:
            y0 = random.randint(0, height - crop)
            x0 = random.randint(0, width - crop)
        sl = np.s_[y0:y0 + crop, x0:x0 + crop]
        x = torch.from_numpy(image[sl][None])
        y = torch.from_numpy(mask[sl][None].astype(np.float32))
        sd = torch.from_numpy(auxiliary[sl][None])
        vm = torch.from_numpy(self.limb[sl][None].astype(np.float32))
        instance_labels = torch.from_numpy(labels[sl].astype(np.int64))
        if self.train:
            if random.random() < 0.5:
                x, y, sd, vm, instance_labels = (
                    x.flip(-1), y.flip(-1), sd.flip(-1), vm.flip(-1), instance_labels.flip(-1)
                )
            if random.random() < 0.5:
                x, y, sd, vm, instance_labels = (
                    x.flip(-2), y.flip(-2), sd.flip(-2), vm.flip(-2), instance_labels.flip(-2)
                )
            turns = random.randrange(4)
            if turns:
                x, y, sd, vm, instance_labels = (
                    torch.rot90(tensor, turns, (-2, -1))
                    for tensor in (x, y, sd, vm, instance_labels)
                )
            if random.random() < 0.3:
                x = (x * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05)).clamp(0, 1)
                x = x ** random.uniform(0.8, 1.2)
        return x, y, sd, vm, instance_labels


def make_embedding_loader(coco, ids, cfg, train):
    dataset = InstanceLabelDataset(coco, ids, cfg, train=train, limb=sol.build_limb_mask(cfg))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
        drop_last=train,
    )
    return loader, dataset


def load_warm_start(model, checkpoint, fold):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["cfg"]["fold"] != fold:
        raise ValueError("Warm-start checkpoint must belong to the same held-out fold")
    result = model.load_state_dict(payload["ema"], strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    allowed_missing = {"head_embed.weight", "head_embed.bias"}
    if unexpected:
        raise ValueError(f"Unexpected warm-start keys: {unexpected}")
    if any(key not in allowed_missing for key in missing):
        raise ValueError(f"Unexpected missing warm-start keys: {missing}")
    return {"missing": missing, "unexpected": unexpected, "cfg": payload["cfg"]}


def make_optimizer(model, base_lr, embed_lr, weight_decay):
    if model.head_embed is None:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    embed_params = list(model.head_embed.parameters())
    embed_ids = {id(parameter) for parameter in embed_params}
    base_params = [parameter for parameter in model.parameters() if id(parameter) not in embed_ids]
    groups = []
    if base_lr > 0:
        groups.append({"params": base_params, "lr": base_lr})
    else:
        for parameter in base_params:
            parameter.requires_grad_(False)
    groups.append({"params": embed_params, "lr": embed_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def predict_full_embeddings(model, image, cfg):
    model.eval()
    height, width = image.shape
    window = cfg.window
    stride = window - cfg.overlap
    probability = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    embeddings = np.zeros((cfg.embedding_dim, height, width), np.float32)
    tta = sol.dihedral_tta_pairs(cfg.tta)
    batch_size = max(1, min(cfg.tta_batch_size, len(tta)))
    y_starts = sorted(set(range(0, max(height - window, 0) + 1, stride)) | {max(height - window, 0)})
    x_starts = sorted(set(range(0, max(width - window, 0) + 1, stride)) | {max(width - window, 0)})
    for y0 in y_starts:
        for x0 in x_starts:
            tile = torch.from_numpy(image[y0:y0 + window, x0:x0 + window][None, None]).to(cfg.device)
            prob_acc = 0.0
            emb_acc = 0.0
            for start in range(0, len(tta), batch_size):
                chunk = tta[start:start + batch_size]
                transformed = torch.cat([augment(tile) for augment, _ in chunk])
                logits, _, batch_embeddings = model.forward_with_embeddings(transformed)
                batch_probability = torch.sigmoid(logits.float())
                for probs, emb, (_, invert) in zip(batch_probability, batch_embeddings.float(), chunk):
                    prob_acc = prob_acc + invert(probs).squeeze(0).cpu().numpy()
                    emb_acc = emb_acc + invert(emb).cpu().numpy()
            prob_acc /= len(tta)
            emb_acc /= len(tta)
            probability[y0:y0 + window, x0:x0 + window] += prob_acc
            embeddings[:, y0:y0 + window, x0:x0 + window] += emb_acc
            count[y0:y0 + window, x0:x0 + window] += 1
    probability = probability / np.maximum(count, 1.0)
    embeddings = embeddings / np.maximum(count[None], 1.0)
    norm = np.linalg.norm(embeddings, axis=0, keepdims=True)
    embeddings = embeddings / np.maximum(norm, 1e-6)
    return probability, embeddings


def predict_embedding_instances(model, image, cfg, limb, parameters):
    probability, embeddings = predict_full_embeddings(model, image, cfg)
    return ie.embedding_to_instances(embeddings, probability, limb, **parameters)


def evaluate_rows(model, dataset, predict_instances, ema=None, partial_path=None, signature=None):
    if ema is not None:
        model = copy.deepcopy(model)
        ema.copy_to(model)
    reports = []
    cached = {}
    if partial_path is not None and partial_path.exists():
        previous = json.loads(partial_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            cached = previous.get("reports_by_image_id", {})
    total = len(dataset.ids)
    for index, image_id in enumerate(dataset.ids, start=1):
        existing = cached.get(str(image_id))
        if existing is not None:
            reports.append(existing)
            print(f"[eval] {index:03d}/{total:03d} reused", flush=True)
            continue
        image, _, _ = dataset._load_full(image_id)
        started = time.perf_counter()
        prediction = predict_instances(model, image, dataset.limb)
        elapsed = time.perf_counter() - started
        report = sol.image_metrics(prediction, sol.ground_truth_instances(dataset, image_id), elapsed_seconds=elapsed)
        report.update(image_id=image_id, file_name=dataset.coco.imgs[image_id]["file_name"])
        reports.append(report)
        cached[str(image_id)] = report
        if partial_path is not None:
            sol._write_json_atomically(partial_path, {"signature": signature, "reports_by_image_id": cached})
        print(
            f"[eval] {index:03d}/{total:03d} PQ {report['pq']:.4f} pred {report['num_pred']} gt {report['num_gt']} elapsed_s {elapsed:.1f}",
            flush=True,
        )
    return reports


def paired_bootstrap(before, after, seed=2026):
    left = {row["image_id"]: row for row in before}
    right = {row["image_id"]: row for row in after}
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical observation IDs")
    keys = sorted(left)
    mean_diff = np.array([right[key]["pq"] - left[key]["pq"] for key in keys], dtype=np.float64)
    before_tp = np.array([left[key]["tp"] for key in keys], dtype=np.float64)
    before_fp = np.array([left[key]["fp"] for key in keys], dtype=np.float64)
    before_fn = np.array([left[key]["fn"] for key in keys], dtype=np.float64)
    before_iou = np.array([sum(left[key]["matched_iou"]) for key in keys], dtype=np.float64)
    after_tp = np.array([right[key]["tp"] for key in keys], dtype=np.float64)
    after_fp = np.array([right[key]["fp"] for key in keys], dtype=np.float64)
    after_fn = np.array([right[key]["fn"] for key in keys], dtype=np.float64)
    after_iou = np.array([sum(right[key]["matched_iou"]) for key in keys], dtype=np.float64)

    def pooled_delta(indices):
        bt = before_tp[indices].sum()
        bf = before_fp[indices].sum()
        bfn = before_fn[indices].sum()
        at = after_tp[indices].sum()
        af = after_fp[indices].sum()
        afn = after_fn[indices].sum()
        bdenom = bt + 0.5 * (bf + bfn)
        adenom = at + 0.5 * (af + afn)
        bpq = before_iou[indices].sum() / bdenom if bdenom else 0.0
        apq = after_iou[indices].sum() / adenom if adenom else 0.0
        return apq - bpq

    rng = np.random.default_rng(seed)
    resamples = rng.integers(len(keys), size=(10000, len(keys)))
    mean_samples = mean_diff[resamples].mean(axis=1)
    pooled_samples = np.array([pooled_delta(sample) for sample in resamples], dtype=np.float64)
    dates = sorted({left[key]["file_name"].split("-")[-1][:8] for key in keys})
    blocks = [np.array([index for index, key in enumerate(keys) if left[key]["file_name"].split("-")[-1][:8] == date])
              for date in dates]
    mean_block = []
    pooled_block = []
    for _ in range(3000):
        picked = np.concatenate([blocks[index] for index in rng.integers(len(blocks), size=len(blocks))])
        mean_block.append(float(mean_diff[picked].mean()))
        pooled_block.append(float(pooled_delta(picked)))
    return {
        "mean_pq_delta": float(mean_diff.mean()),
        "dataset_pq_delta": float(pooled_delta(np.arange(len(keys)))),
        "observation_bootstrap_95ci": {
            "mean_pq": np.quantile(mean_samples, [0.025, 0.975]).tolist(),
            "dataset_pq": np.quantile(pooled_samples, [0.025, 0.975]).tolist(),
        },
        "date_block_bootstrap_95ci": {
            "mean_pq": np.quantile(mean_block, [0.025, 0.975]).tolist(),
            "dataset_pq": np.quantile(pooled_block, [0.025, 0.975]).tolist(),
        },
        "n_observations": len(keys),
        "n_dates": len(dates),
        "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty.",
    }


def fit_arm(root, arm, coco, train_ids, monitor_ids, cfg):
    work = root / arm
    work.mkdir(parents=True, exist_ok=True)
    history_path = work / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) == cfg.epochs:
            return history, json.loads((work / "warm_start.json").read_text(encoding="utf-8"))
        raise RuntimeError("Interrupted training preserved; use a new experiment directory")
    started = work / "started.json"
    if started.exists():
        if (work / "fold1_best.pt").exists() or history_path.exists():
            raise RuntimeError("Interrupted training preserved; use a new experiment directory")
        started.unlink()
    sol._write_json_atomically(started, vars(cfg))
    sol.seed_everything(cfg.seed + cfg.fold)
    train_loader, _ = make_embedding_loader(coco, train_ids, cfg, True)
    monitor_dataset = sol.FilamentDataset(coco, monitor_ids, cfg, False, sol.build_limb_mask(cfg))
    model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
    warm_start = load_warm_start(model, PARENT, cfg.fold)
    sol._write_json_atomically(work / "warm_start.json", warm_start)
    settings = ARM_SETTINGS[arm]
    optimizer = make_optimizer(model, settings["base_lr"], settings["embed_lr"], cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device == "cuda")
    ema = sol.EMA(model, cfg.ema_decay)
    best = -1.0
    history = []

    def monitor_rows():
        if arm == "control":
            parameters = control_parameters()
            scorer_cfg = replace(cfg, **parameters)
            return evaluate_rows(
                model,
                monitor_dataset,
                lambda current, image, limb: sol.prob_to_instances(sol.predict_full([current], image, scorer_cfg), scorer_cfg, limb),
                ema,
            )
        return evaluate_rows(
            model,
            monitor_dataset,
            lambda current, image, limb: predict_embedding_instances(current, image, cfg, limb, EMBED_DEFAULT),
            ema,
        )

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started_epoch = time.monotonic()
        running = 0.0
        for batch_index, (x, y, sd, vm, instance_labels) in enumerate(train_loader):
            x, y, sd, vm, instance_labels = (
                tensor.to(cfg.device, non_blocking=True)
                for tensor in (x, y, sd, vm, instance_labels)
            )
            with torch.amp.autocast("cuda", enabled=cfg.device == "cuda"):
                loss, _ = sol.criterion(model, x, y, sd, vm, cfg, instance_labels=instance_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, batch {batch_index}")
            group_start = (batch_index // cfg.grad_accum_steps) * cfg.grad_accum_steps
            group_size = min(cfg.grad_accum_steps, len(train_loader) - group_start)
            scaler.scale(loss / group_size).backward()
            if (batch_index + 1) % cfg.grad_accum_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            running += loss.item()
        scheduler.step()
        rows = monitor_rows()
        summary = ex.summarize(rows)
        if summary["mean_pq"] > best:
            best = summary["mean_pq"]
            checkpoint = {
                "model": model.state_dict(),
                "ema": {key: value.detach().cpu() for key, value in ema.m.items()},
                "cfg": vars(cfg),
                "epoch": epoch,
                "completed_epochs": epoch + 1,
                "best_monitor_pq": best,
                "arm": arm,
            }
            temporary = work / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(work / "fold1_best.pt")
        row = {
            "epoch": epoch + 1,
            "loss": running / len(train_loader),
            "monitor_pq": summary["mean_pq"],
            "monitor_dataset_pq": summary["dataset_pq"],
            "seconds": time.monotonic() - started_epoch,
        }
        history.append(row)
        sol._write_json_atomically(history_path, history)
        sol._write_json_atomically(work / f"monitor_epoch{epoch + 1:03d}.json", {"summary": summary, "rows": rows})
        print(f"{arm} {row}", flush=True)
    return history, warm_start


def load_best_model(path, cfg):
    model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["ema"], strict=True)
    return model


def main(smoke=False):
    torch.set_num_threads(4)
    root = ROOT / "smoke_v2" if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(root, smoke)
    coco, train_ids, validation_ids = sol.get_fold(replace(cfg, limit=0))
    split = ex.partition(validation_ids, cfg.seed)
    train_keys = {sol.base_name(coco.imgs[image_id]["file_name"]) for image_id in train_ids}
    validation_keys = {sol.base_name(coco.imgs[image_id]["file_name"]) for image_id in validation_ids}
    assert not train_keys & validation_keys
    if smoke:
        selected = set(sorted(train_keys)[:8])
        train_ids = [image_id for image_id in train_ids if sol.base_name(coco.imgs[image_id]["file_name"]) in selected]
        split = {name: ids[:2] for name, ids in split.items()}
    protocol = {
        "config": vars(cfg),
        "train_ids": train_ids,
        "split": split,
        "smoke": smoke,
        "arms": ARM_SETTINGS,
        "control_parameters": control_parameters(),
        "embedding_default": EMBED_DEFAULT,
        "embedding_grid": EMBED_GRID,
        "checkpoint_sha256": digest(PARENT),
        "source_sha256": {
            name: digest(name)
            for name in ("solution.py", "instance_embeddings.py", "experiment_instance_embeddings.py", cfg.train_json)
        },
        "selection": "Both arms train on the same fold-1 training records. Monitor uses 16 observations; candidate calibration uses 40; paired audit uses 85.",
        "gate": f"Full assessment only if embedding best monitor mean PQ improves over control by more than {MONITOR_MIN_DELTA:.3f}.",
    }
    protocol = json.loads(json.dumps(protocol))
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Inputs changed; use a new experiment directory")
    sol._write_json_atomically(protocol_path, protocol)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"status": "training", "arms": {}}
    if state.get("status") == "complete":
        print("Completed experiment preserved.", flush=True)
        return

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            REPORT.write_text(
                "# Instance-embedding experiment\n\n"
                "Two warm-started arms share the same fold-1 physical-observation split and monitor/calibration/audit protocol. "
                "The control arm keeps semantic-component extraction; the candidate arm adds discriminative embeddings and clusters foreground pixels in embedding space. "
                "The parent Round-5 checkpoint is loaded with `strict=False`; only the new embedding head starts unmatched, while existing weights are preserved.\n\n"
                "Run `python experiment_instance_embeddings.py --smoke`, then `python experiment_instance_embeddings.py`. "
                "The 85-observation audit is only used in full mode after the monitor gate passes. No test labels or test images are used.\n\n"
                "```json\n" + json.dumps(state, indent=2) + "\n```\n",
                encoding="utf-8",
            )

    try:
        histories = {}
        warm_starts = {}
        for arm, settings in ARM_SETTINGS.items():
            arm_cfg = replace(cfg, work_dir=str(root / arm), w_embed=settings["w_embed"], embedding_dim=settings["embedding_dim"])
            state.update(status="training", active_arm=arm)
            save()
            histories[arm], warm_starts[arm] = fit_arm(root, arm, coco, train_ids, split["monitor"], arm_cfg)
            state["arms"][arm] = histories[arm]
            state.setdefault("warm_start", {})[arm] = warm_starts[arm]
            save()
            gc.collect()
            torch.cuda.empty_cache()
        control_best = max(row["monitor_pq"] for row in histories["control"])
        embedding_best = max(row["monitor_pq"] for row in histories["embedding"])
        state["monitor"] = {"control": control_best, "embedding": embedding_best, "delta": embedding_best - control_best}
        gate_passed = smoke or embedding_best > control_best + MONITOR_MIN_DELTA
        state["decision"] = "monitor_gate_passed" if gate_passed else "rejected_monitor"
        if not gate_passed:
            state.update(status="complete", promoted=False, leaderboard_score=None)
            save()
            print(json.dumps(state, indent=2), flush=True)
            return

        state.update(status="calibrating")
        save()
        candidate_cfg = replace(cfg, embedding_dim=ARM_SETTINGS["embedding"]["embedding_dim"], w_embed=ARM_SETTINGS["embedding"]["w_embed"])
        control_cfg = replace(cfg, embedding_dim=ARM_SETTINGS["control"]["embedding_dim"], w_embed=0.0)
        candidate_model = load_best_model(root / "embedding" / "fold1_best.pt", candidate_cfg)
        control_model = load_best_model(root / "control" / "fold1_best.pt", control_cfg)
        calibration_dataset = sol.FilamentDataset(coco, split["calibration"], cfg, False, sol.build_limb_mask(cfg))
        grid = []
        for setting in EMBED_GRID:
            rows = evaluate_rows(
                candidate_model,
                calibration_dataset,
                lambda current, image, limb, current_setting=setting: predict_embedding_instances(
                    current, image, candidate_cfg, limb, current_setting
                ),
                partial_path=root / f"calibration_{setting['threshold']}_{setting['min_area']}_{setting['eps']}_{setting['min_samples']}.json",
                signature=setting,
            )
            grid.append({"parameters": setting, "summary": ex.summarize(rows)})
        selected = max(grid, key=lambda row: row["summary"]["mean_pq"])
        state["selection"] = selected
        sol._write_json_atomically(root / "selection.json", {"selected": selected, "grid": grid})
        save()

        state.update(status="assessing")
        save()
        audit_dataset = sol.FilamentDataset(coco, split["audit"], cfg, False, sol.build_limb_mask(cfg))
        control_rows = evaluate_rows(
            control_model,
            audit_dataset,
            lambda current, image, limb: sol.prob_to_instances(sol.predict_full([current], image, control_cfg), control_cfg, limb),
            partial_path=root / "audit_control_partial.json",
            signature={"arm": "control", "parameters": control_parameters()},
        )
        candidate_rows = evaluate_rows(
            candidate_model,
            audit_dataset,
            lambda current, image, limb: predict_embedding_instances(current, image, candidate_cfg, limb, selected["parameters"]),
            partial_path=root / "audit_embedding_partial.json",
            signature={"arm": "embedding", "parameters": selected["parameters"]},
        )
        assessment = {
            "control": ex.summarize(control_rows),
            "candidate": ex.summarize(candidate_rows),
            "paired": paired_bootstrap(control_rows, candidate_rows),
            "selected": selected["parameters"],
            "smoke": smoke,
        }
        sol._write_json_atomically(root / "assessment.json", {**assessment, "control_rows": control_rows, "candidate_rows": candidate_rows})
        state["assessment"] = assessment
        state["promoted"] = bool(
            assessment["candidate"]["mean_pq"] > assessment["control"]["mean_pq"]
            and assessment["paired"]["observation_bootstrap_95ci"]["mean_pq"][0] > 0
        )
        state.update(status="complete", leaderboard_score=None)
        save()
        print(json.dumps(state, indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)