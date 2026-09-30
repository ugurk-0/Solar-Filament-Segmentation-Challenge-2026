"""Calibrate seeded mask growth on 40 observations, then assess on 85."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import solution as sol
import pq_experiment as ex
from instance_quality import matrix_metrics, summarize
from seeded_instances import seeded_instances

ROOT = Path('runs/seeded_instances_20260930')
BASE = Path('runs/pq_refinement_20260928_fixed')
PARENT = Path('runs/pq_research_20260926/training_round5/fold1_best.pt')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(smoke=False):
    torch.set_num_threads(4)
    root = ROOT / 'smoke' if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    cfg = replace(ex.config(root), thresh=.8, min_area=500, close_kernel=1)
    coco, _, ids = sol.get_fold(cfg)
    split = ex.partition(ids, cfg.seed)
    if smoke:
        split = {key: values[:2] for key, values in split.items()}
    _, dataset = sol.make_loader(coco, ids, cfg, False)
    previous = json.loads((BASE / 'protocol.json').read_text())
    signature = digest(PARENT)
    assert signature == previous['checkpoint_sha256']
    for key in ('tta', 'window', 'overlap', 'img_size', 'limb_geometry', 'limb_max_deg', 'disk_radius'):
        assert getattr(cfg, key) == previous['config'][key], key
    cache = Path('runs/pq_research_20260926/comparison_round5') / ('probabilities_' + signature[:12])
    candidates = [None] + [dict(low=low, high=.8, min_seed=500, min_area=500, separate=separate)
                          for low in (.5, .65, .75) for separate in (False, True)]
    protocol = dict(checkpoint_sha256=signature, config=vars(cfg), split=split,
                    candidates=candidates, smoke=smoke,
                    source_sha256={name: digest(name) for name in
                                   ('solution.py', 'seeded_instances.py', 'experiment_seeded_instances.py', cfg.train_json)},
                    selection='40 calibration observations only; assessment reused and exploratory')
    protocol = json.loads(json.dumps(protocol))
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Inputs changed; use a new experiment directory')
    sol._write_json_atomically(path, protocol)
    if (root / 'assessment.json').exists():
        print('Completed assessment preserved.')
        return

    def evaluate(image_id, settings):
        probability = np.load(cache / f'{image_id}.npy', allow_pickle=False)
        assert probability.shape == dataset.limb.shape and np.isfinite(probability).all()
        assert probability.min() >= 0 and probability.max() <= 1
        gt = sol.ground_truth_instances(dataset, image_id)
        rows = []
        for setting in settings:
            masks = (sol.prob_to_instances(probability, cfg, dataset.limb) if setting is None else
                     seeded_instances(probability, dataset.limb, **setting))
            row = matrix_metrics(sol.instance_iou_matrix(masks, gt))
            row.update(image_id=image_id, file_name=coco.imgs[image_id]['file_name'])
            rows.append(row)
        return rows

    calibration = [[] for _ in candidates]
    for number, image_id in enumerate(split['calibration'], 1):
        for target, row in zip(calibration, evaluate(image_id, candidates)):
            target.append(row)
        print(f'calibration {number}/{len(split["calibration"])}', flush=True)
    grid = [dict(parameters=setting, summary=summarize(rows)) for setting, rows in zip(candidates, calibration)]
    selected = max(grid, key=lambda entry: entry['summary']['mean_pq'])
    sol._write_json_atomically(root / 'selection.json', dict(selected=selected, grid=grid))
    print('Frozen selection:', selected, flush=True)
    before, after = [], []
    for number, image_id in enumerate(split['audit'], 1):
        a, b = evaluate(image_id, [None, selected['parameters']])
        before.append(a)
        after.append(b)
        if number % 10 == 0:
            print(f'assessment {number}/{len(split["audit"])}', flush=True)
    baseline, candidate = summarize(before), summarize(after)
    if not smoke:
        historical = json.loads((BASE / 'assessment.json').read_text())['after']
        assert abs(baseline['dataset_pq'] - historical['dataset_pq']) < 1e-9
    paired = ex.paired_bootstrap(before, after)
    promoted = bool(not smoke and candidate['dataset_pq'] > baseline['dataset_pq']
                    and paired['observation_bootstrap_95ci'][0] > 0)
    report = dict(baseline=baseline, candidate=candidate, paired=paired, promoted=promoted,
                  selected=selected['parameters'], smoke=smoke, leaderboard_score=None)
    sol._write_json_atomically(root / 'assessment.json', dict(**report, baseline_rows=before, selected_rows=after))
    if not smoke:
        Path('reports/seeded_instances_20260930.md').write_text(
            '# Seeded mask-growth experiment\n\n'
            'Hypothesis: growing confident components into weaker connected foreground improves shapes '
            'without admitting unseeded false detections. Compare merging versus preserving confident seeds. '
            'The frozen Round-5 model, no-TTA inference, observation split and support mask are unchanged.\n\n'
            'Seven settings including the baseline were compared on 40 calibration observations. '
            'The selected setting was frozen before evaluating 85 reused research observations. '
            'Repeated assessment is exploratory; these are not leaderboard scores.\n\n'
            'Run `python experiment_seeded_instances.py --smoke`, then `python experiment_seeded_instances.py`. '
            'No neural-network weights are retrained. The retained baseline and submissions are preserved.\n\n'
            '```json\n' + json.dumps(report, indent=2) + '\n```\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
