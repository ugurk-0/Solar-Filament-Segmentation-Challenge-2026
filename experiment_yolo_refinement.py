"""Calibration-gated boundary refinement using cached YOLO and U-Net outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as maskutils
import solution as sol
import pq_experiment as ex
from experiment_yolo import digest, ROOT as YOLO_ROOT
from instance_quality import matrix_metrics, summarize
from yolo_refinement import refine_masks

ROOT = Path('runs/yolo_refinement_20261006')
GRID = [dict(radius=r, threshold=t, confidence=c, min_area=128)
        for r in (8, 24) for t in (.5, .8) for c in (.1, .3)]


def main(smoke=False):
    torch.set_num_threads(4)
    root, parent = (ROOT / 'smoke', YOLO_ROOT / 'smoke') if smoke else (ROOT, YOLO_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    parent_state = json.loads((parent / 'state.json').read_text())
    if parent_state['status'] != 'complete':
        raise RuntimeError('Complete the direct YOLO experiment first')
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    signature = digest(parent / 'best_pq.pt')
    cache = parent / ('predictions_' + signature[:12])
    unet = json.loads(Path('runs/pq_refinement_20260928_fixed/protocol.json').read_text())
    if digest(unet['checkpoint']) != unet['checkpoint_sha256']:
        raise ValueError('Retained U-Net checkpoint changed')
    probability_cache = Path('runs/pq_research_20260926/comparison_round5') / ('probabilities_' + unet['checkpoint_sha256'][:12])
    protocol = dict(smoke=smoke, grid=GRID, split=parent_protocol['split'],
                    yolo_sha256=signature, unet_sha256=unet['checkpoint_sha256'],
                    calibration_gate='mean PQ > direct YOLO + .005 and pooled PQ > direct YOLO',
                    source_sha256={name: digest(name) for name in
                                   ('experiment_yolo_refinement.py', 'yolo_refinement.py', 'yolo_data.py', 'solution.py')})
    protocol_path = root / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Refinement inputs changed; use a new output directory')
    sol._write_json_atomically(protocol_path, protocol)
    state_path = root / 'state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status') == 'complete':
        print('Completed refinement experiment preserved.')
        return
    cfg = ex.config(root)
    coco, _, ids = sol.get_fold(cfg)
    _, dataset = sol.make_loader(coco, ids, cfg, False)

    def score(selected_ids, settings):
        rows = [[] for _ in settings]
        for number, image_id in enumerate(selected_ids, 1):
            with np.load(cache / f'{image_id}.npz') as stored:
                masks = np.unpackbits(stored['masks'], axis=-1, count=cfg.img_size).astype(bool)
                scores = stored['scores']
            probability = np.load(probability_cache / f'{image_id}.npy')
            gt = sol.ground_truth_instances(dataset, image_id)
            gt_rle = [maskutils.encode(np.asfortranarray(m, dtype=np.uint8)) for m in gt]
            for output, parameters in zip(rows, settings):
                pred = refine_masks(masks, scores, probability, dataset.limb, **parameters)
                pred_rle = [maskutils.encode(np.asfortranarray(m, dtype=np.uint8)) for m in pred]
                iou = (maskutils.iou(pred_rle, gt_rle, [0] * len(gt)) if pred and gt
                       else np.zeros((len(pred), len(gt)), np.float64))
                row = matrix_metrics(iou)
                row.update(image_id=image_id, file_name=coco.imgs[image_id]['file_name'])
                output.append(row)
            print(f'refinement {number}/{len(selected_ids)}', flush=True)
        return rows

    state = dict(status='calibrating', grid=GRID)

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            Path('reports/yolo_refinement_20261006.md').write_text(
                '# YOLO-guided U-Net boundary refinement\n\n'
                'No additional training: intersect each dilated YOLO instance with retained U-Net '
                'probabilities, then resolve overlap by YOLO confidence. Eight predeclared settings '
                'are calibrated on 40 observations. Only a calibration winner over direct YOLO '
                'advances to the reused 85-observation assessment. No test labels used.\n\n'
                '```json\n' + json.dumps(state, indent=2) + '\n```\n', encoding='utf-8')

    save()
    rows = score(protocol['split']['calibration'], GRID)
    grid = [dict(parameters=p, summary=summarize(r)) for p, r in zip(GRID, rows)]
    selected = max(grid, key=lambda item: item['summary']['mean_pq'])
    baseline = parent_state['selected']['summary']
    passed = (selected['summary']['mean_pq'] > baseline['mean_pq'] + .005
              and selected['summary']['dataset_pq'] > baseline['dataset_pq'])
    state.update(grid=grid, selected=selected, direct_yolo_calibration=baseline, calibration_passed=passed)
    sol._write_json_atomically(root / 'selection.json', state)
    if passed:
        state['status'] = 'assessing'
        save()
        rows = score(protocol['split']['audit'], [selected['parameters']])[0]
        measured = summarize(rows)
        retained = json.loads((parent / 'assessment.json').read_text())
        paired = ex.paired_bootstrap(retained['selected_rows'], rows) if not smoke else None
        promoted = bool(not smoke and measured['dataset_pq'] > retained['summary']['dataset_pq']
                        and paired['observation_bootstrap_95ci'][0] > 0)
        state.update(assessment=measured, paired=paired, promoted=promoted,
                     decision='promoted_research_candidate' if promoted else 'not_promoted')
        sol._write_json_atomically(root / 'assessment.json', dict(summary=measured, paired=paired, selected_rows=rows))
    else:
        state.update(promoted=False, decision='rejected_calibration')
    state.update(status='complete', leaderboard_score=None)
    if smoke:
        state['decision'] = 'smoke_only'
    save()
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
