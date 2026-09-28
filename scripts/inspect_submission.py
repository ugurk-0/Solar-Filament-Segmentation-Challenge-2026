"""Decode and audit a submission CSV without running inference or using test labels."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as maskutils


def inspect(csv_path, test_dir, manifest_path, output_dir):
    csv_path, test_dir, output_dir = map(Path, (csv_path, test_dir, output_dir))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Expected exactly filament_id,segmentation_rle")
        rows = list(reader)
    groups, ids = defaultdict(list), set()
    for row in rows:
        identifier = row["filament_id"]
        if identifier in ids:
            raise ValueError(f"Duplicate ID: {identifier}")
        ids.add(identifier)
        stem, suffix = identifier.rsplit("_", 1)
        if not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(f"Invalid instance suffix: {identifier}")
        groups[stem + ".jpeg"].append(row)
    images = {path.name: path for path in test_dir.glob("*.jpeg")}
    if not images:
        raise ValueError("No test JPEGs found")
    if set(groups) - images.keys():
        raise ValueError("CSV references unknown images")
    expected_counts = manifest["image_instance_counts"]
    if set(expected_counts) != set(images):
        raise ValueError("Manifest and test image inventory differ")
    if manifest["n_images"] != len(images) or manifest["n_instances"] != len(rows):
        raise ValueError("Manifest totals disagree with CSV/image inventory")
    areas, lengths, examples = [], [], []
    figure_data = None
    for name, path in sorted(images.items()):
        image_rows = groups[name]
        if len(image_rows) != expected_counts[name]:
            raise ValueError(f"Manifest count mismatch for {name}")
        if [int(row["filament_id"].rsplit("_", 1)[1]) for row in image_rows] != list(range(1, len(image_rows) + 1)):
            raise ValueError(f"Nonconsecutive instance numbering for {name}")
        with Image.open(path) as image:
            width, height = image.size
        occupied = np.zeros((height, width), dtype=bool)
        labels = np.zeros((height, width), dtype=np.uint16) if figure_data is None else None
        first_mask = None
        for index, row in enumerate(image_rows, 1):
            rle = {"size": [height, width], "counts": row["segmentation_rle"].encode("ascii")}
            mask = maskutils.decode(rle).astype(bool)
            if mask.shape != occupied.shape or not mask.any():
                raise ValueError(f"Invalid/empty mask: {row['filament_id']}")
            if np.any(occupied & mask):
                raise ValueError(f"Overlapping masks: {row['filament_id']}")
            occupied |= mask
            encoded = maskutils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")
            if encoded != row["segmentation_rle"]:
                raise ValueError(f"RLE round trip differs: {row['filament_id']}")
            areas.append(int(mask.sum()))
            lengths.append(len(encoded))
            if labels is not None:
                labels[mask] = index
                if first_mask is None:
                    first_mask = mask.copy()
                    examples.append({"filament_id": row["filament_id"],
                                     "rle_prefix": encoded[:80], "rle_characters": len(encoded),
                                     "area_pixels": areas[-1],
                                     "bbox_xywh": maskutils.toBbox(rle).astype(int).tolist(),
                                     "shape": [height, width]})
        if labels is not None and image_rows:
            figure_data = (path, labels, first_mask)
    if not rows:
        raise ValueError("No predicted instances to inspect")
    report = {
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "images": len(images), "images_with_predictions": sum(bool(groups[name]) for name in images),
        "images_without_predictions": sum(not groups[name] for name in images),
        "rows": len(rows), "unique_ids": len(ids),
        "instances_per_image": {"min": min(expected_counts.values()), "max": max(expected_counts.values())},
        "area_pixels": {"min": min(areas), "median": float(np.median(areas)), "max": max(areas)},
        "rle_characters": {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths)},
        "all_masks_decoded": True, "all_rle_round_trips_match": True,
        "overlapping_instances": 0, "manifest_counts_match": True,
        "example": examples[0], "scope": "Serialization and consistency audit; no ground-truth or quality evaluation."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if figure_data:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path, labels, mask = figure_data
        with Image.open(path) as image:
            pixels = np.asarray(image)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        axes[0].imshow(pixels, cmap="gray")
        axes[0].set_title("Test image: " + path.stem)
        axes[1].imshow(pixels, cmap="gray")
        axes[1].imshow(np.ma.masked_where(labels == 0, labels), cmap="tab20", alpha=0.8,
                       interpolation="nearest", vmin=1, vmax=max(int(labels.max()), 2))
        axes[1].set_title(f"{labels.max()} CSV rows = {labels.max()} instances")
        axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("First row decoded: " + examples[0]["filament_id"])
        for axis in axes:
            axis.axis("off")
        fig.savefig(output_dir / "csv_explained.png", dpi=130)
        plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.csv, args.test_dir, args.manifest, args.output_dir), indent=2))
