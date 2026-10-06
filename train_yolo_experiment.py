"""Verified YOLO training/fine-tuning with a fixed observation split and PQ gate."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import torch
import ultralytics
from ultralytics import YOLO
import solution as sol
import pq_experiment as ex
from experiment_yolo import digest, evaluate
from instance_quality import summarize
from yolo_data import prepare_dataset
from yolo_model_identity import model_identity
from yolo_training import SupportSegmentationTrainer


def main(args):
    if ultralytics.__version__ != '8.3.253':
        raise RuntimeError('Use requirements-yolo.txt in the separate YOLO environment')
    torch.set_num_threads(4)
    root = Path(args.output) / 'smoke' if args.smoke else Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ex.config(root)
    coco, train, validation = sol.get_fold(cfg)
    split = ex.partition(validation, cfg.seed)
    if args.smoke:
        keys = sorted({sol.base_name(coco.imgs[i]['file_name']) for i in train})[:4]
        train = [i for i in train if sol.base_name(coco.imgs[i]['file_name']) in keys]
        split = {key: ids[:2] for key, ids in split.items()}
    size, epochs = (640, 1) if args.smoke else (args.imgsz, args.epochs)
    initial = Path(args.weights) if args.weights else Path(args.output) / 'pretrained' / f'yolo11{args.variant}-seg.pt'
    initial.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(initial))
    identity = model_identity(model, args.variant)
    baseline = Path(args.baseline)
    baseline_summary = json.loads((baseline / 'assessment.json').read_text())['summary']
    protocol = dict(variant=args.variant, actual_model=identity, init_sha256=digest(initial),
                    baseline=str(baseline), baseline_sha256=digest(baseline / 'assessment.json'),
                    split=split, train_ids=train, imgsz=size, epochs=epochs, batch=args.batch,
                    lr0=args.lr0, nominal_batch=8, smoke=args.smoke,
                    mode='fine_tune_reset_optimizer' if args.weights else 'pretrained_training',
                    selection='Monitor PQ at confidence .1 / area 128; calibration grid unchanged; no TTA.',
                    stopping='At least 6 epochs; stop after 5 without >.001 monitoring gain.',
                    source_sha256={name: digest(name) for name in
                        ('train_yolo_experiment.py', 'experiment_yolo.py', 'yolo_model_identity.py',
                         'yolo_data.py', 'yolo_training.py', 'yolo_prediction.py', 'solution.py',
                         'scripts/update_yolo_readme.py', cfg.train_json)})
    protocol_path, state_path = root / 'protocol.json', root / 'state.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Inputs changed; choose a new --output directory')
    sol._write_json_atomically(protocol_path, protocol)
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='preparing', history=[])
    if state['status'] == 'complete':
        print('Completed run preserved.')
        return
    report = Path('reports') / (root.name + '.md')

    def save():
        sol._write_json_atomically(state_path, state)
        if not args.smoke:
            report.write_text(
                f'# YOLO11{args.variant} follow-up experiment\n\n'
                f'Actual loaded model: {identity}. Initial checkpoint: `{initial}`. '
                f'Input {size}, batch {args.batch}, nominal batch 8, learning rate {args.lr0}, '
                f'budget {epochs} epochs. Reset optimizer for fine-tuning. '
                'Same grouped split, support-aware loss, no geometric augmentation or TTA. '
                'The parent checkpoint is retained unless monitoring PQ improves. '
                'Calibrate on 40 observations, then compare on the reused 85-observation assessment '
                f'against `{baseline.name}` (mean {baseline_summary["mean_pq"]:.4f}, '
                f'pooled {baseline_summary["dataset_pq"]:.4f}). '
                'Promotion requires higher pooled PQ and paired mean-PQ confidence interval above zero.\n\n'
                '```json\n' + json.dumps(state, indent=2) + '\n```\n', encoding='utf-8')

    try:
        save()
        data_dir = root / 'dataset' if args.smoke or not args.dataset_cache else Path(args.dataset_cache)
        data = prepare_dataset(coco, train, split['monitor'], cfg, data_dir)
        _, dataset = sol.make_loader(coco, validation, cfg, False)
        selected_checkpoint = root / 'best_pq.pt'
        if not state.get('training_complete'):
            if (root / 'training/weights/last.pt').exists():
                raise RuntimeError('Interrupted checkpoint preserved; choose a new run or explicitly resume it')
            # Prediction may fuse Conv/BN in-place. Never fuse the weights that
            # will initialize training; evaluate a separate model instance.
            control_model = YOLO(str(initial))
            control = summarize(evaluate(control_model, dataset, split['monitor'], coco, size,
                                [dict(confidence=.1, min_area=128)], confidence_floor=.1)[0])
            del control_model
            gc.collect()
            torch.cuda.empty_cache()
            state.update(status='training', initial_monitor=control, actual_model=identity)
            best = [control['mean_pq'] if args.weights else -1.]
            last_improved, started = [0], time.monotonic()
            if args.weights:
                shutil.copyfile(initial, selected_checkpoint)
                state['best_epoch'] = 0
            save()

            def monitor(trainer):
                epoch = trainer.epoch + 1
                if state['history'] and state['history'][-1]['epoch'] >= epoch:
                    return
                candidate = YOLO(str(trainer.last))
                state['trained_model_identity'] = model_identity(candidate, args.variant)
                measured = summarize(evaluate(candidate, dataset, split['monitor'], coco, size,
                                      [dict(confidence=.1, min_area=128)], confidence_floor=.1)[0])
                if measured['mean_pq'] > best[0]:
                    if measured['mean_pq'] > best[0] + .001:
                        last_improved[0] = epoch
                    best[0] = measured['mean_pq']
                    shutil.copyfile(trainer.last, selected_checkpoint)
                    state['best_epoch'] = epoch
                state['history'].append(dict(epoch=epoch, monitor=measured,
                                             elapsed_seconds=time.monotonic() - started))
                save()
                print(f'PQ epoch={epoch}: {measured["mean_pq"]:.5f}; best={best[0]:.5f}', flush=True)
                del candidate
                gc.collect()
                torch.cuda.empty_cache()
                if not args.smoke and epoch >= 6 and epoch - last_improved[0] >= 5:
                    trainer.stop = True

            model.add_callback('on_fit_epoch_end', monitor)
            model.train(trainer=SupportSegmentationTrainer, data=str(data.resolve()), epochs=epochs,
                        imgsz=size, batch=args.batch, nbs=8, workers=0, cache=False, device=0,
                        optimizer='AdamW', lr0=args.lr0, lrf=.05, warmup_epochs=1,
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
        if args.weights and not args.smoke and state.get('best_epoch') == 0:
            state.update(status='complete', promoted=False, decision='no_monitor_gain', leaderboard_score=None)
            save()
            return
        model = YOLO(str(selected_checkpoint))
        model_identity(model, args.variant)
        signature = digest(selected_checkpoint)
        cache = root / ('predictions_' + signature[:12])
        cache.mkdir(exist_ok=True)
        settings = [dict(confidence=c, min_area=a) for c in (.05, .1, .2, .3, .5) for a in (32, 128, 500)]
        state['status'] = 'calibrating'
        save()
        rows = evaluate(model, dataset, split['calibration'], coco, size, settings, cache, confidence_floor=.05)
        grid = [dict(parameters=p, summary=summarize(r)) for p, r in zip(settings, rows)]
        selected = max(grid, key=lambda item: item['summary']['mean_pq'])
        sol._write_json_atomically(root / 'selection.json', dict(selected=selected, grid=grid, checkpoint_sha256=signature))
        state.update(status='assessing', selected=selected)
        save()
        rows = evaluate(model, dataset, split['audit'], coco, size, [selected['parameters']], cache, confidence_floor=.05)[0]
        measured = summarize(rows)
        reference = json.loads((baseline / 'assessment.json').read_text())
        paired = ex.paired_bootstrap(reference['selected_rows'], rows) if not args.smoke else None
        promoted = bool(not args.smoke and measured['dataset_pq'] > reference['summary']['dataset_pq']
                        and paired['observation_bootstrap_95ci'][0] > 0)
        state.update(status='complete', assessment=measured, paired=paired, promoted=promoted,
                     decision='smoke_only' if args.smoke else 'promoted_research_candidate' if promoted else 'not_promoted',
                     leaderboard_score=None)
        sol._write_json_atomically(root / 'assessment.json', dict(summary=measured, paired=paired, selected_rows=rows))
        save()
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise
    finally:
        if args.update_readme and state.get('status') in ('complete', 'failed'):
            from scripts.update_yolo_readme import update_readme
            update_readme()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--variant', choices=['n', 's', 'm'], default='n')
    result.add_argument('--weights', help='Existing trained checkpoint for fine-tuning with a fresh optimizer')
    result.add_argument('--output', required=True)
    result.add_argument('--dataset-cache', default='runs/yolo11n_20261003/dataset')
    result.add_argument('--baseline', default='runs/yolo11n_20261003')
    result.add_argument('--imgsz', type=int, default=1536)
    result.add_argument('--epochs', type=int, default=12)
    result.add_argument('--batch', type=int, default=2)
    result.add_argument('--lr0', type=float, default=.00015)
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--update-readme', action='store_true', help='Refresh tracked result rows when the run finishes')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
