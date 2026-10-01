"""Inventory actual competition labels and leakage-safe learning opportunities."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from matplotlib.path import Path as PolygonPath
import pq_experiment as ex
import solution as sol


def main():
    cfg = ex.config('runs/data_learning_20261001')
    coco, train_ids, val_ids = sol.get_fold(cfg)
    groups = defaultdict(list)
    for image_id, info in coco.imgs.items():
        groups[sol.base_name(info['file_name'])].append(image_id)
    train_keys = {sol.base_name(coco.imgs[i]['file_name']) for i in train_ids}
    val_keys = {sol.base_name(coco.imgs[i]['file_name']) for i in val_ids}
    assert not train_keys & val_keys
    anns = coco.loadAnns(coco.getAnnIds(imgIds=train_ids))
    malformed, out_of_bounds, total_points, inside_points = 0, 0, 0, 0
    lengths, vertex_counts = [], []
    for ann in anns:
        points = np.asarray(ann.get('spine', []), dtype=float)
        if points.size < 4 or points.size % 2 or not np.isfinite(points).all():
            malformed += 1
            continue
        points = points.reshape(-1, 2)
        info = coco.imgs[ann['image_id']]
        out_of_bounds += int(np.any((points[:, 0] < 0) | (points[:, 0] >= info['width']) |
                                   (points[:, 1] < 0) | (points[:, 1] >= info['height'])))
        vertex_counts.append(len(points))
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
        inside = np.zeros(len(points), bool)
        for polygon in ann['segmentation']:
            inside |= PolygonPath(np.asarray(polygon).reshape(-1, 2)).contains_points(points, radius=1e-6)
        total_points += len(points)
        inside_points += int(inside.sum())
    quantiles = lambda values: dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                                       np.quantile(values, [0, .25, .5, .75, 1]).tolist()))
    counts = Counter(a['image_id'] for a in anns)
    report = dict(image_records=len(coco.imgs), physical_observations=len(groups), annotations=len(coco.anns),
                  annotation_sets_per_observation=dict(sorted(Counter(map(len, groups.values())).items())),
                  train_records=len(train_ids), train_observations=len(train_keys),
                  validation_observations=len(val_keys), train_validation_overlap=0,
                  training_annotations=len(anns), categories={str(k): v['name'] for k,v in coco.cats.items()},
                  training_category_counts=dict(Counter(a['category_id'] for a in anns)),
                  training_area_pixels=quantiles([a['area'] for a in anns]),
                  training_annotations_below_500_pixels=sum(a['area'] < 500 for a in anns),
                  training_instances_per_record=quantiles([counts[i] for i in train_ids]),
                  spine=dict(malformed=malformed, outside_image=out_of_bounds,
                             vertex_count=quantiles(vertex_counts), length_pixels=quantiles(lengths),
                             vertices_inside_own_polygon_fraction=inside_points / total_points),
                  training_stations=dict(Counter(key[-2:] for key in train_keys)),
                  test_stations=dict(Counter(p.stem[-2:] for p in Path('MAGFiLO_1.0_Kaggle_2026/test/test_images').glob('*.jpeg'))),
                  training_years=dict(sorted(Counter(key[:4] for key in train_keys).items())),
                  caveat='Geometry quality and label distributions are measured only on fold-1 training annotations; records are not independent observations.')
    sol._write_json_atomically(Path('reports/learning_data_audit_20261001.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
