"""Pretrained YOLO11m-seg with support-masked loss and PQ-based selection."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
import solution as sol
import pq_experiment as ex
from instance_quality import matrix_metrics, summarize
from yolo_data import prepare_dataset, disjoint_masks
from yolo_training import SupportSegmentationTrainer
from yolo_prediction import ChunkedSegmentationPredictor
from yolo_model_identity import model_identity

ROOT = Path('runs/yolo11m_20261003')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction(model, dataset, image_id, imgsz):
    image = sol.load_image(Path(dataset.cfg.data_dir) / 'train_images' / dataset.coco.imgs[image_id]['file_name'])
    image = np.rint(image * dataset.limb * 255).astype(np.uint8)
    result = model.predict(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), imgsz=imgsz, conf=.01,
                           iou=.7, max_det=100, retina_masks=True, device=0, half=True,
                           verbose=False, save=False, predictor=ChunkedSegmentationPredictor)[0]
    if result.masks is None:
        return np.empty((0, *image.shape), bool), np.empty(0, np.float32)
    masks = result.masks.data.cpu().numpy().astype(bool)
    scores = result.boxes.conf.cpu().numpy()
    assert masks.shape[1:] == image.shape
    return masks, scores


def evaluate(model, dataset, ids, coco, imgsz, parameters, cache=None):
    rows = [[] for _ in parameters]
    for number, image_id in enumerate(ids, 1):
        cache_file = cache / f'{image_id}.npz' if cache else None
        if cache_file and cache_file.exists():
            with np.load(cache_file) as stored:
                masks = np.unpackbits(stored['masks'], axis=-1, count=dataset.limb.shape[1]).astype(bool)
                scores = stored['scores']
        else:
            masks, scores = prediction(model, dataset, image_id, imgsz)
            if cache_file:
                np.savez_compressed(cache_file, masks=np.packbits(masks, axis=-1), scores=scores)
        gt = sol.ground_truth_instances(dataset, image_id)
        for target, setting in zip(rows, parameters):
            pred = disjoint_masks(masks, scores, dataset.limb, **setting)
            row = matrix_metrics(sol.instance_iou_matrix(pred, gt))
            row.update(image_id=image_id, file_name=coco.imgs[image_id]['file_name'])
            target.append(row)
        if len(ids) > 16 and number % 10 == 0:
            print(f'PQ evaluation {number}/{len(ids)}', flush=True)
    return rows


def main(smoke=False):
    if ultralytics.__version__ != '8.3.253':
        raise RuntimeError('Use pinned requirements-yolo.txt')
    torch.set_num_threads(4)
    root = ROOT / 'smoke' if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    cfg = ex.config(root)
    coco, train, validation = sol.get_fold(cfg)
    split = ex.partition(validation, cfg.seed)
    if smoke:
        keys = sorted({sol.base_name(coco.imgs[i]['file_name']) for i in train})[:4]
        train = [i for i in train if sol.base_name(coco.imgs[i]['file_name']) in keys]
        split = {key: values[:2] for key, values in split.items()}
    imgsz, epochs = (640, 1) if smoke else (1536, 50)
    protocol = dict(smoke=smoke, train_ids=train, split=split, imgsz=imgsz, epochs=epochs, batch=1,
                    ultralytics=ultralytics.__version__, model='yolo11m-seg.pt',
                    selection='Epoch: mean PQ on 16 monitor observations at confidence .1, area 128. Calibration: 40. Assessment: 85 reused.',
                    stopping='At least 8 epochs, stop after 6 epochs without monitor PQ improvement >0.001; maximum 50.',
                    source_sha256={name: digest(name) for name in
                                   ('solution.py', 'yolo_data.py', 'yolo_training.py', 'yolo_prediction.py', 'experiment_yolo11m.py', cfg.train_json)})
    protocol = json.loads(json.dumps(protocol))
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Inputs changed; use a new output directory')
    sol._write_json_atomically(path, protocol)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='preparing', history=[])
    if state['status'] == 'complete':
        print('Completed experiment preserved.')
        return

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            Path('reports/yolo11m_20261003.md').write_text(
                '# YOLO11m instance-segmentation experiment (50 epochs)\n\n'
                'Pretrained medium model, 1536-pixel full disk, batch 1 and nominal batch 8, AMP, '
                'one actual annotation set per training observation. No geometric/mosaic augmentation; '
                'photometric value augmentation only. Separate masks, support-aware assignment, classification '
                'and mask losses. Native-coordinate output masks, confidence-ordered non-overlap ownership. '
                'All chirality categories map to one filament class.\n\n'
                'Epoch selected by PQ on monitoring images, not YOLO mAP. Confidence/area selected on '
                '40 calibration observations; 85-observation assessment is reused research data. '
                'Reference is the promoted skeleton-merge candidate (mean 0.2891, pooled 0.2878). '
                'No test labels or leaderboard optimization.\n\n'
                'Run `python experiment_yolo.py --smoke`, then `python experiment_yolo.py`, '
                'in the environment from requirements-yolo.txt.\n\n'
                '```json\n' + json.dumps(state, indent=2) + '\n```\n', encoding='utf-8')
    try:
        save()
        data = prepare_dataset(coco, train, split['monitor'], cfg, root / 'dataset')
        _, dataset = sol.make_loader(coco, validation, cfg, False)
        pq_checkpoint = root / 'best_pq.pt'
        if not state.get('training_complete'):
            state['status'] = 'training'
            save()
            weights = ROOT / 'pretrained/yolo11m-seg.pt'
            weights.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(weights))
            state['actual_model'] = model_identity(model, 'm')
            state['pretrained_sha256'] = digest(weights)
            best, last_improved, started = [-1.0], [0], time.monotonic()

            def monitor(trainer):
                # on_fit_epoch_end also runs after final validation; avoid duplicate epochs.
                epoch = trainer.epoch + 1
                if state['history'] and state['history'][-1]['epoch'] >= epoch:
                    return
                evaluated = YOLO(str(trainer.last))
                rows = evaluate(evaluated, dataset, split['monitor'], coco, imgsz,
                                [dict(confidence=.1, min_area=128)])[0]
                measured = summarize(rows)
                if measured['mean_pq'] > best[0]:
                    if measured['mean_pq'] > best[0] + .001:
                        last_improved[0] = epoch
                    best[0] = measured['mean_pq']
                    shutil.copyfile(trainer.last, pq_checkpoint)
                    state['best_epoch'] = epoch
                state['history'].append(dict(epoch=epoch, monitor=measured,
                                             elapsed_seconds=time.monotonic() - started))
                save()
                print(f'PQ MONITOR epoch={epoch} mean={measured["mean_pq"]:.5f} best={best[0]:.5f}', flush=True)
                del evaluated
                gc.collect()
                torch.cuda.empty_cache()
                if not smoke and epoch >= 8 and epoch - last_improved[0] >= 6:
                    trainer.stop = True
            model.add_callback('on_fit_epoch_end', monitor)
            previous = root / 'training/weights/last.pt'
            if previous.exists():
                raise RuntimeError('Interrupted training retained; do not overwrite it')
            model.train(trainer=SupportSegmentationTrainer, data=str(data.resolve()), epochs=epochs,
                        imgsz=imgsz, batch=1, nbs=8, workers=0, cache=False, device=0,
                        optimizer='AdamW', lr0=.001, lrf=.05, warmup_epochs=1,
                        cos_lr=True, amp=True, deterministic=True, seed=2026,
                        mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0,
                        scale=0, shear=0, perspective=0, flipud=0, fliplr=0, multi_scale=False,
                        hsv_h=0, hsv_s=0, hsv_v=.15, overlap_mask=False, mask_ratio=4,
                        project=str(root.resolve()), name='training', exist_ok=False,
                        patience=0, plots=False, save=True, verbose=False)
            state['training_complete'] = True
            save()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        state['status'] = 'calibrating'
        save()
        model = YOLO(str(pq_checkpoint))
        signature = digest(pq_checkpoint)
        cache = root / ('predictions_' + signature[:12])
        cache.mkdir(exist_ok=True)
        settings = [dict(confidence=confidence, min_area=area)
                    for confidence in (.05, .1, .2, .3, .5) for area in (32, 128, 500)]
        rows = evaluate(model, dataset, split['calibration'], coco, imgsz, settings, cache)
        grid = [dict(parameters=p, summary=summarize(r)) for p,r in zip(settings, rows)]
        selected = max(grid, key=lambda entry: entry['summary']['mean_pq'])
        sol._write_json_atomically(root / 'selection.json', dict(selected=selected, grid=grid, checkpoint_sha256=signature))
        state.update(status='assessing', selected=selected)
        save()
        rows = evaluate(model, dataset, split['audit'], coco, imgsz, [selected['parameters']], cache)[0]
        baseline_path = Path('runs/skeleton_merge_20261003/assessment.json')
        baseline = json.loads(baseline_path.read_text())
        # The previous run records per-image rows alongside its aggregate comparison.
        baseline_rows = baseline.get('merged_rows', baseline.get('selected_rows'))
        if not smoke and baseline_rows is None:
            raise ValueError('Missing merged baseline rows for paired comparison')
        measured = summarize(rows)
        paired = ex.paired_bootstrap(baseline_rows, rows) if not smoke else None
        promoted = bool(not smoke and measured['dataset_pq'] > baseline['merged']['dataset_pq']
                        and paired['observation_bootstrap_95ci'][0] > 0)
        state.update(status='complete', assessment=measured, paired=paired, promoted=promoted,
                     decision='smoke_only' if smoke else 'promoted_research_candidate' if promoted else 'not_promoted',
                     leaderboard_score=None)
        sol._write_json_atomically(root / 'assessment.json', dict(summary=measured, paired=paired, selected_rows=rows))
        save()
        print(json.dumps(state, indent=2), flush=True)
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)

