"""Full-training-fold matched comparison of observation sampling and spine loss."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import solution as sol
import pq_experiment as ex
from instance_quality import summarize
from spine_learning import ObservationDataset, auxiliary_loss

ROOT = Path('runs/spine_learning_20261001')
PARENT = Path('runs/pq_research_20260926/training_round5/fold1_best.pt')
BASE = Path('runs/pq_refinement_20260928_fixed')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fit(coco, records, monitor, cfg, save_progress):
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    history_path = work / 'history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if len(history) == cfg.epochs:
            return history
        raise RuntimeError('Interrupted training preserved; use a new output directory')
    if (work / 'started.json').exists():
        if (work / 'fold1_best.pt').exists():
            raise RuntimeError('Interrupted training preserved; use a new output directory')
        (work / 'started.json').unlink()
    sol._write_json_atomically(work / 'started.json', vars(cfg))
    sol.seed_everything(cfg.seed + cfg.fold)
    valid = sol.build_limb_mask(cfg)
    trainset = ObservationDataset(coco, records, cfg, True, valid)
    valset = sol.FilamentDataset(coco, monitor, cfg, False, valid)
    loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True,
                                       num_workers=0, pin_memory=cfg.device == 'cuda')
    model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
    payload = torch.load(PARENT, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['ema'])
    del payload
    spine_params = list(model.head_spine.parameters())
    spine_ids = {id(p) for p in spine_params}
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in spine_ids], lr=cfg.lr),
        dict(params=spine_params, lr=1e-3)], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.device == 'cuda')
    ema = sol.EMA(model, cfg.ema_decay)
    history, best = [], -1.0
    for epoch in range(cfg.epochs):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3)
        print(f'{work.name} epoch {epoch + 1}/{cfg.epochs}: {len(trainset)} observations', flush=True)
        for index, batch in enumerate(loader):
            x, target, spine, support = [tensor.to(cfg.device, non_blocking=True) for tensor in batch]
            with torch.amp.autocast('cuda', enabled=cfg.device == 'cuda'):
                outputs = model(x)
                segmentation, _ = sol.criterion(lambda _: outputs, x, target, spine, support, cfg)
                extra = auxiliary_loss(outputs[1], spine, support) if cfg.aux_spine_weight else segmentation.new_zeros(())
                loss = segmentation + cfg.aux_spine_weight * extra
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            group_start = index // cfg.grad_accum_steps * cfg.grad_accum_steps
            group_size = min(cfg.grad_accum_steps, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            if (index + 1) % cfg.grad_accum_steps == 0 or index + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            totals += [loss.item(), segmentation.item(), extra.item()]
            if (index + 1) % 100 == 0:
                print(f'{work.name}: {index + 1}/{len(loader)}', flush=True)
        scheduler.step()
        report = sol.evaluate_detailed(model, valset, cfg, ema)
        pq = report['aggregate']['pq']['mean']
        if pq > best:
            best = pq
            checkpoint = dict(ema={k: v.detach().cpu() for k,v in ema.m.items()}, cfg=vars(cfg),
                              epoch=epoch, completed_epochs=epoch + 1, best_pq=best,
                              training_method='uniform observation sampling; optional auxiliary spine heatmap',
                              inference='standard solution.py, raw grayscale, same segmentation head')
            temporary = work / 'best.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(work / 'fold1_best.pt')
            del checkpoint
        row = dict(epoch=epoch + 1, mean_loss=totals[0] / len(loader),
                   segmentation_loss=totals[1] / len(loader), spine_loss=totals[2] / len(loader),
                   monitor_pq=pq, seconds=time.monotonic() - started)
        history.append(row)
        sol._write_json_atomically(history_path, history)
        sol._write_json_atomically(work / f'validation_epoch{epoch + 1}.json', report)
        save_progress(history)
        print(row, flush=True)
    return history


def main(smoke=False):
    torch.set_num_threads(4)
    root = ROOT / 'smoke' if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    cfg = replace(ex.config(root), epochs=2 if smoke else 3, limit=8 if smoke else 0,
                  lr=1e-5, ema_decay=.99, thresh=.8, min_area=500, close_kernel=1,
                  oversample_p=.5, pos_weight_cap=1, w_spine=0, preview_count=0)
    coco, records, validation = sol.get_fold(replace(cfg, limit=0))
    split = ex.partition(validation, cfg.seed)
    train_keys = {sol.base_name(coco.imgs[i]['file_name']) for i in records}
    assert not train_keys & {sol.base_name(coco.imgs[i]['file_name']) for i in validation}
    if smoke:
        selected = set(sorted(train_keys)[:8])
        records = [i for i in records if sol.base_name(coco.imgs[i]['file_name']) in selected]
        split = {key: values[:2] for key, values in split.items()}
    protocol = dict(config=vars(cfg), train_records=records, split=split, smoke=smoke,
                    arms={'observation_control': 0.0, 'spine_auxiliary': .1},
                    checkpoint_sha256=digest(PARENT),
                    source_sha256={name: digest(name) for name in
                                   ('solution.py', 'spine_learning.py', 'experiment_spine_learning.py', cfg.train_json)},
                    selection='Epoch and arm by monitoring PQ. Best arm assessed only if monitor gain >0.005 over parent. Assessment reused; exploratory.',
                    checkpoints='Best EMA weights only; completed arms reusable; interrupted training requires new directory')
    protocol = json.loads(json.dumps(protocol))
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Inputs changed; use new experiment directory')
    sol._write_json_atomically(path, protocol)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='training', arms={})
    if state['status'] == 'complete':
        print('Completed experiment preserved.')
        return

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            Path('reports/spine_learning_20261001.md').write_text(
                '# Observation-balanced and spine-supervised learning\n\n'
                'Both arms use all 566 training observations, one sampled real annotation set per visit, '
                'three epochs, same Round-5 initialization/seed/crops/loss/post-processing. '
                'The auxiliary arm adds a 0.1-weight balanced MSE on annotated spine heatmaps. '
                'One distance transform per image; support/foreground clipping; aligned augmentations. '
                'A faster learning rate is used for the previously unused auxiliary head in both arms. '
                'No annotation-derived input features are required at inference.\n\n'
                'Run `python experiment_spine_learning.py --smoke`, then `python experiment_spine_learning.py`. '
                'Checkpoint/arm selection uses 16 monitor observations. Only the selected arm may advance '
                'to the reused 85-observation assessment. No test labels are used.\n\n'
                '```json\n' + json.dumps(state, indent=2) + '\n```\n', encoding='utf-8')
    try:
        for arm, weight in protocol['arms'].items():
            arm_cfg = replace(cfg, work_dir=str(root / arm))
            arm_cfg.aux_spine_weight = weight
            state.update(status='training', active_arm=arm)
            save()
            def progress(history):
                state['arms'][arm] = history
                save()
            state['arms'][arm] = fit(coco, records, split['monitor'], arm_cfg, progress)
            gc.collect()
            torch.cuda.empty_cache()
            save()
        parent = json.loads(Path('runs/autoupgrade_20260928/baseline_monitor.json').read_text())['aggregate']['pq']['mean']
        best_by_arm = {arm: max(r['monitor_pq'] for r in history) for arm, history in state['arms'].items()}
        selected = max(best_by_arm, key=best_by_arm.get)
        state.update(parent_monitor_pq=parent, best_monitor=best_by_arm, selected=selected,
                     decision='smoke_only' if smoke else 'rejected_monitor', promoted=False)
        if not smoke and best_by_arm[selected] > parent + .005:
            state.update(status='assessing', decision='assessment_pending')
            save()
            model = sol.FilamentUNet(cfg, pretrained=False).to(cfg.device)
            payload = torch.load(root / selected / 'fold1_best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(payload['ema'])
            del payload
            _, dataset = sol.make_loader(coco, split['audit'], cfg, False)
            report = sol.evaluate_detailed(model, dataset, cfg)
            baseline = json.loads((BASE / 'assessment.json').read_text())
            paired = ex.paired_bootstrap(baseline['selected_rows'], report['per_image'])
            summary = summarize(report['per_image'])
            sol._write_json_atomically(root / 'assessment.json', dict(report=report, paired=paired))
            state['assessment'] = dict(summary=summary, paired=paired)
            state['promoted'] = bool(summary['dataset_pq'] > baseline['after']['dataset_pq'] and paired['observation_bootstrap_95ci'][0] > 0)
            state['decision'] = 'promoted_research_candidate' if state['promoted'] else 'rejected_assessment'
        state.update(status='complete', leaderboard_score=None)
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
