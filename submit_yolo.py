"""Export the retained, calibrated YOLO model to a checked competition CSV.

Reads only test images and saved model/selection metadata. No training labels,
calibration sweep, or leaderboard-driven adjustment is performed by this command.
"""
import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

import solution as sol
from yolo_data import disjoint_masks
from yolo_model_identity import model_identity
from yolo_prediction import ChunkedSegmentationPredictor


# Fingerprints of the evaluated inference helpers, captured before retiring unused
# training branches. Training-only edits cannot invalidate the export contract.
SOLUTION_FUNCTION_SHA256 = {'build_limb_mask': '30f91e16a07ba59f2446d77ff0e3c0b711b634f16b8af6a3489e9d37f0220cc9', 'load_image': '1e246af105667e1b0a198911d3fbcaf1ed0a1cf7433dc4e328af5c78bc81d3d7', '_write_json_atomically': 'cc5af1574093eac7913aac3829f8e327d959f1bac03fab0bf9abfaed2c915451', 'mask_to_rle': 'd762f47de8fd6263c0c75b768e1111134619adfb3de40a4d88250193919204c9'}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(checkpoint, run_dir, test_dir, output, expected_images=180):
    checkpoint, run_dir, test_dir, output = map(Path, (checkpoint, run_dir, test_dir, output))
    if ultralytics.__version__ != "8.3.253":
        raise RuntimeError("Use requirements-yolo.txt (ultralytics==8.3.253)")
    if not torch.cuda.is_available():
        raise RuntimeError("This validated inference configuration requires CUDA with half precision")
    selection_path, protocol_path = run_dir / "selection.json", run_dir / "protocol.json"
    identity_path = run_dir / "model_identity.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if protocol.get("smoke"):
        raise ValueError("Smoke experiments cannot be exported as final submissions")
    checkpoint_sha256 = digest(checkpoint)
    if checkpoint_sha256 != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the recorded model identity")
    if protocol["ultralytics"] != ultralytics.__version__:
        raise ValueError("Inference package differs from the evaluated package")
    # These own preprocessing, support and native mask reconstruction. Preserve
    # their recorded bytes instead of silently using unassessed inference code.
    for name in ("yolo_data.py", "yolo_prediction.py"):
        expected = protocol["source_sha256"].get(name)
        if expected and digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Evaluated inference source changed: {name}")
    tree = ast.parse((Path(__file__).parent / "solution.py").read_text(encoding="utf-8"))
    actual_functions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                        for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name in SOLUTION_FUNCTION_SHA256}
    if actual_functions != SOLUTION_FUNCTION_SHA256:
        raise ValueError("Evaluated image loading, support, RLE or manifest helper changed")
    parameters = selection["parameters"]
    if set(parameters) != {"confidence", "min_area"}:
        raise ValueError("Unexpected calibrated post-processing schema")
    files = sorted(test_dir.glob("*.jpeg"))
    if len(files) != expected_images or not files:
        raise ValueError(f"Expected {expected_images} test JPEGs; found {len(files)}")
    if len({path.stem for path in files}) != len(files):
        raise ValueError("Test image stems must be unique")
    if (output / "submission.csv").exists() or (output / "submission_manifest.json").exists():
        raise FileExistsError(f"Completed submission is preserved; choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = sol.Cfg(img_size=2048, disk_radius=950, limb_geometry="longitude", limb_max_deg=70.0)
    valid = sol.build_limb_mask(cfg)
    model = YOLO(str(checkpoint))
    actual_identity = model_identity(model, identity["scale"])
    if actual_identity["parameters"] != identity["parameters"]:
        raise ValueError("Loaded model parameter count differs from the retained model")
    rows, counts, image_hashes = [], {}, {}
    started = time.perf_counter()
    inference = dict(imgsz=int(protocol["imgsz"]), confidence_floor=0.01,
                     nms_iou=0.7, max_det=100, retina_masks=True,
                     device=0, half=True, tta=False, **parameters)
    for number, path in enumerate(files, 1):
        image = sol.load_image(path)
        if image.shape != (2048, 2048) or not np.isfinite(image).all():
            raise ValueError(f"Expected finite native 2048x2048 image: {path.name}")
        image_hashes[path.name] = digest(path)
        prepared = np.rint(image * valid * 255).astype(np.uint8)
        result = model.predict(
            cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR), imgsz=inference["imgsz"],
            conf=inference["confidence_floor"], iou=inference["nms_iou"],
            max_det=inference["max_det"], retina_masks=True, device=0, half=True,
            verbose=False, save=False, predictor=ChunkedSegmentationPredictor)[0]
        if result.masks is None:
            masks = np.empty((0, *valid.shape), dtype=bool)
            scores = np.empty(0, dtype=np.float32)
        else:
            native = result.masks.data.cpu().numpy()
            if native.shape[1:] != valid.shape or not np.isin(native, (0, 1)).all():
                raise ValueError(f"Expected binary native-resolution masks: {path.name}")
            masks = native.astype(bool)
            scores = result.boxes.conf.cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite detection scores: {path.name}")
        instances = disjoint_masks(masks, scores, valid, **parameters)
        occupied = np.zeros_like(valid)
        for index, mask in enumerate(instances, 1):
            if mask.dtype != np.bool_ or mask.shape != valid.shape or not mask.any():
                raise ValueError(f"Invalid instance: {path.stem}_{index}")
            if np.any(mask & occupied) or np.any(mask & ~valid):
                raise ValueError(f"Overlap or unsupported pixels: {path.stem}_{index}")
            occupied |= mask
            encoded = sol.mask_to_rle(mask)
            if not encoded or any(char in encoded for char in ("'", '"', "\r", "\n")):
                raise ValueError("RLE must contain raw compressed counts without quote wrappers")
            decoded = sol.maskutils.decode({"size": [2048, 2048], "counts": encoded.encode("ascii")})
            if not np.array_equal(decoded.astype(bool), mask):
                raise ValueError(f"RLE round-trip failed: {path.stem}_{index}")
            rows.append(dict(filament_id=f"{path.stem}_{index}", segmentation_rle=encoded))
        counts[path.name] = len(instances)
        sol._write_json_atomically(output / "progress.json", dict(
            status="running", images_complete=number, expected_images=len(files),
            last_image=path.name, instances=len(rows), elapsed_seconds=time.perf_counter() - started))
        print(f"{number}/{len(files)} {path.name}: {len(instances)} instances", flush=True)
        del result, masks, scores, instances
    ids = [row["filament_id"] for row in rows]
    if len(ids) != len(set(ids)) or sum(counts.values()) != len(rows):
        raise ValueError("Duplicate instance IDs or mismatched image counts")
    csv_path = output / "submission.csv"
    temporary = output / "submission.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    with temporary.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"] or list(reader) != rows:
            raise ValueError("CSV write/read verification failed")
    temporary.replace(csv_path)
    description = ("YOLO11n-seg, epoch 17 selected by monitoring PQ; confidence 0.50 and minimum "
                   "area 128 calibrated on 40 observations. 1536-pixel input, native 2048 masks, "
                   "solar annotation support, confidence-ordered non-overlap ownership; no TTA. "
                   "Local reused 85-observation assessment: mean PQ 0.3199, pooled PQ 0.3319. "
                   "COCO RLE and image coverage validated. Kaggle score pending.")
    if len(description) > 500:
        raise ValueError("Submission description exceeds 500 characters")
    (output / "submission_description.txt").write_text(description + "\n", encoding="utf-8")
    manifest = dict(solution_function_sha256=actual_functions, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha256,
                    model=actual_identity, inference=inference,
                    support=dict(img_size=cfg.img_size, radius=cfg.disk_radius,
                                 geometry=cfg.limb_geometry, limb_max_deg=cfg.limb_max_deg),
                    selection_sha256=digest(selection_path), protocol_sha256=digest(protocol_path),
                    source_sha256={name: digest(Path(__file__).parent / name) for name in
                                   ("submit_yolo.py", "solution.py", "yolo_data.py", "yolo_prediction.py")},
                    image_sha256=image_hashes, image_instance_counts=counts,
                    n_images=len(files), n_instances=len(rows), csv_sha256=digest(csv_path),
                    validation=dict(csv_schema="passed", unique_ids="passed", native_binary_masks="passed",
                                    image_coverage="passed", rle_round_trip="all passed", instance_overlap="none"),
                    elapsed_seconds=time.perf_counter() - started, kaggle_submission="not_uploaded")
    sol._write_json_atomically(output / "submission_manifest.json", manifest)
    sol._write_json_atomically(output / "progress.json", dict(status="complete", images_complete=len(files),
                                                            instances=len(rows), elapsed_seconds=manifest["elapsed_seconds"]))
    print(f"Saved {csv_path}: {len(rows)} instances, SHA256 {manifest['csv_sha256']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/yolo11n_20261003"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-dir", type=Path, default=Path("MAGFiLO_1.0_Kaggle_2026/test/test_images"))
    parser.add_argument("--output", type=Path, default=Path("submissions/final_20261108"))
    parser.add_argument("--expected-images", type=int, default=180)
    args = parser.parse_args()
    export(args.checkpoint or args.run_dir / "best_pq.pt", args.run_dir, args.test_dir,
           args.output, args.expected_images)


if __name__ == "__main__":
    main()
