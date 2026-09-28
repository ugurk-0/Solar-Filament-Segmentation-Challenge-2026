import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as maskutils
import pytest


def fixture_export(tmp_path, overlap=False):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("inspect_submission", root / "scripts/inspect_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("image_a", "image_b"):
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(image_dir / f"{name}.jpeg")
    mask = np.zeros((8, 8), np.uint8)
    mask[2:4, 3:5] = 1
    counts = maskutils.encode(np.asfortranarray(mask))["counts"].decode("ascii")
    rows = [["image_a_1", counts]]
    if overlap:
        rows.append(["image_a_2", counts])
    csv_path = tmp_path / "submission.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"n_images": 2, "n_instances": len(rows),
                                   "image_instance_counts": {"image_a.jpeg": len(rows), "image_b.jpeg": 0}}))
    return module, csv_path, image_dir, manifest


def test_audit_counts_images_without_predictions_and_decodes_geometry(tmp_path):
    module, csv_path, images, manifest = fixture_export(tmp_path)
    result = module.inspect(csv_path, images, manifest, tmp_path / "audit")
    assert result["images"] == 2
    assert result["images_with_predictions"] == 1
    assert result["images_without_predictions"] == 1
    assert result["example"]["bbox_xywh"] == [3, 2, 2, 2]
    assert result["example"]["area_pixels"] == 4
    assert (tmp_path / "audit/csv_explained.png").is_file()


def test_audit_rejects_overlapping_instances(tmp_path):
    module, csv_path, images, manifest = fixture_export(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="Overlapping"):
        module.inspect(csv_path, images, manifest, tmp_path / "audit")
