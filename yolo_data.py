"""Prepare support-masked, observation-grouped YOLO instance labels."""
import json
from pathlib import Path
import cv2
import numpy as np
import solution as sol


def mask_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
    if not contours:
        raise ValueError('Nonempty instance has no valid polygon')
    if len(contours) == 1:
        polygon = contours[0]
    else:
        from ultralytics.data.converter import merge_multi_segment
        polygon = np.concatenate(merge_multi_segment([c.reshape(-1) for c in contours]), axis=0)
    reconstructed = np.zeros_like(mask, np.uint8)
    cv2.fillPoly(reconstructed, [polygon.astype(np.int32)], 1)
    iou = (reconstructed.astype(bool) & mask).sum() / (reconstructed.astype(bool) | mask).sum()
    if iou < .98:
        raise ValueError(f'Polygon round-trip IoU too low: {iou:.6f}')
    return polygon.astype(np.float32), float(iou)


def disjoint_masks(masks, scores, valid, confidence=.1, min_area=128):
    """Keep each predicted instance identity; assign overlap to higher confidence."""
    if len(masks) != len(scores):
        raise ValueError('Mask/score count mismatch')
    occupied = np.zeros_like(valid, bool)
    result = []
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        if scores[index] < confidence:
            continue
        mask = np.asarray(masks[index], bool)
        if mask.shape != valid.shape:
            raise ValueError('Predicted masks must have native image coordinates')
        mask = mask & valid & ~occupied
        if mask.sum() >= min_area:
            result.append(mask)
            occupied |= mask
    return result


def prepare_dataset(coco, training, monitoring, cfg, output):
    output = Path(output)
    manifest_path = output / 'manifest.json'
    grouped = {}
    for image_id in training:
        grouped.setdefault(sol.base_name(coco.imgs[image_id]['file_name']), image_id)
    training = list(grouped.values())
    assert not set(grouped) & {sol.base_name(coco.imgs[i]['file_name']) for i in monitoring}
    contract = dict(train_ids=training, monitor_ids=monitoring, radius=cfg.disk_radius,
                    support=cfg.limb_geometry, support_angle=cfg.limb_max_deg)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['contract'] != contract:
            raise ValueError('Dataset inputs changed; use a new dataset directory')
        return output / 'dataset.yaml'
    valid = sol.build_limb_mask(cfg)
    ious, instances = [], 0
    for split, ids in [('train', training), ('val', monitoring)]:
        images, labels = output / 'images' / split, output / 'labels' / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        dataset = sol.FilamentDataset(coco, ids, cfg, False, valid)
        for number, image_id in enumerate(ids, 1):
            image, _, _ = dataset._load_full(image_id)
            image = np.rint(image * valid * 255).astype(np.uint8)
            if not cv2.imwrite(str(images / f'{image_id}.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError('Could not write YOLO image')
            rows = []
            for mask in sol.ground_truth_instances(dataset, image_id):
                polygon, iou = mask_polygon(mask)
                polygon /= np.array([mask.shape[1], mask.shape[0]], np.float32)
                assert np.isfinite(polygon).all() and polygon.min() >= 0 and polygon.max() <= 1
                rows.append('0 ' + ' '.join(f'{v:.8f}' for v in polygon.ravel()))
                ious.append(iou)
                instances += 1
            (labels / f'{image_id}.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            if number % 50 == 0:
                print(f'prepare {split}: {number}/{len(ids)}', flush=True)
    (output / 'dataset.yaml').write_text(
        f'path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n', encoding='utf-8')
    sol._write_json_atomically(manifest_path, dict(contract=contract, instances=instances,
                              min_polygon_roundtrip_iou=min(ious), mean_polygon_roundtrip_iou=float(np.mean(ious)),
                              annotation_choice='First record per physical observation; no synthetic consensus',
                              classes='All chirality labels map to filament; predictions retain separate masks'))
    return output / 'dataset.yaml'
