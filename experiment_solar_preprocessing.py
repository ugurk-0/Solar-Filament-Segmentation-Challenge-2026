"""Matched raw-versus-radial-corrected fine-tuning pilot; no test-set tuning."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import solution as sol
import pq_experiment as ex
from instance_quality import summarize
from solar_preprocessing import radial_correct

ROOT = Path('runs/solar_preprocessing_20260930')
PARENT = Path('runs/pq_research_20260926/training_round5/fold1_best.pt')
BASE = Path('runs/pq_refinement_20260928_fixed')
OriginalDataset = sol.FilamentDataset


class SolarDataset(OriginalDataset):
    def _load_full(self, image_id):
        if image_id in self._cache:
            return self._cache[image_id]
        image, mask, auxiliary = super()._load_full(image_id)
        strength = getattr(self.cfg, 'radial_strength', 0)
        if strength:
            image = radial_correct(image, self.cfg.disk_radius, strength)
        result = image, mask, auxiliary
        self._cache[image_id] = result
        return result


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(smoke=False):
    torch.set_num_threads(4)
    root = ROOT / 'smoke' if smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    cfg = replace(ex.config(root), thresh=.8, min_area=500, close_kernel=1,
                  epochs=2, lr=1e-5, ema_decay=.99, eval_every=1,
                  eval_max_images=16, init_ckpt=str(PARENT), pos_weight_cap=1,
                  oversample_p=.5, preview_count=0, limit=4 if smoke else 0)
    original_fold = sol.get_fold
    coco, train, validation = original_fold(replace(cfg, limit=0))
    groups = {}
    for image_id in train:
        groups.setdefault(sol.base_name(coco.imgs[image_id]['file_name']), image_id)
    keys = sorted(groups)
    chosen = np.random.default_rng(cfg.seed).choice(keys, min(4 if smoke else 128, len(keys)), replace=False)
    selected = [groups[key] for key in chosen]
    assert not set(chosen) & {sol.base_name(coco.imgs[i]['file_name']) for i in validation}
    splits = ex.partition(validation, cfg.seed)
    protocol = dict(config=vars(cfg), training_ids=selected, split=splits,
                    arms={'raw_control': 0, 'radial_corrected': .5}, smoke=smoke,
                    checkpoint_sha256=digest(PARENT),
                    source_sha256={name: digest(name) for name in
                                   ('solution.py', 'solar_preprocessing.py', 'experiment_solar_preprocessing.py', cfg.train_json)},
                    selection='Two epochs per arm; checkpoint by 16 monitor observations; assess correction only if it beats parent and raw control by 0.005')
    protocol = json.loads(json.dumps(protocol))
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Inputs changed; use a new output directory')
    sol._write_json_atomically(path, protocol)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='training', arms={})
    if state['status'] == 'complete':
        print('Completed pilot preserved.')
        return

    def save():
        sol._write_json_atomically(state_path, state)
        if not smoke:
            Path('reports/solar_preprocessing_20260930.md').write_text(
                '# Radial-correction matched fine-tuning pilot\n\n'
                'Hypothesis: bounded illumination correction improves limb filament detection. '
                '128 training observations, two epochs per arm, same parent, crop schedule, seed, loss, '
                'support and post-processing. This pilot does not establish the optimum training budget. '
                'A 0.005 monitoring gain over both parent and raw control is required before assessing '
                'the corrected model on 85 reused research observations. No test labels are used.\n\n'
                'Reproduce: `python experiment_solar_preprocessing.py --smoke`, then '
                '`python experiment_solar_preprocessing.py`. Corrected checkpoints require SolarDataset '
                'or radial_correct before inference; the ordinary CLI does not apply that transform.\n\n'
                '```json\n' + json.dumps(state, indent=2) + '\n```\n', encoding='utf-8')

    def fixed_fold(current_cfg):
        if current_cfg.fold != cfg.fold:
            raise ValueError('Unexpected fold')
        return coco, selected, splits['monitor'][:2] if smoke else validation

    try:
        sol.get_fold, sol.FilamentDataset = fixed_fold, SolarDataset
        for arm, strength in protocol['arms'].items():
            arm_cfg = replace(cfg, work_dir=str(root / arm))
            arm_cfg.radial_strength = strength
            history_path = Path(arm_cfg.work_dir) / 'history.json'
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            state.update(status='training', active_arm=arm)
            save()
            if len(history) < cfg.epochs:
                def finished(epoch, checkpoint):
                    state['arms'][arm] = json.loads(history_path.read_text())
                    save()
                sol.train(arm_cfg, epoch_callback=finished)
            history = json.loads(history_path.read_text())
            state['arms'][arm] = [{key: row[key] for key in ('epoch', 'loss', 'pq', 'elapsed_seconds')} for row in history]
            gc.collect()
            torch.cuda.empty_cache()
            save()
        parent_monitor = json.loads(Path('runs/autoupgrade_20260928/baseline_monitor.json').read_text())['aggregate']['pq']['mean']
        raw = max(row['pq'] for row in state['arms']['raw_control'])
        corrected = max(row['pq'] for row in state['arms']['radial_corrected'])
        state.update(parent_monitor_pq=parent_monitor, raw_monitor_pq=raw, corrected_monitor_pq=corrected,
                     decision='smoke_only' if smoke else 'rejected_monitor', promoted=False)
        if not smoke and corrected > max(raw, parent_monitor) + .005:
            state.update(status='assessing', decision='assessment_pending')
            save()
            assess_cfg = replace(cfg, work_dir=str(root / 'radial_corrected'))
            assess_cfg.radial_strength = .5
            dataset = SolarDataset(coco, splits['audit'], assess_cfg, False, sol.build_limb_mask(assess_cfg))
            model = sol.FilamentUNet(assess_cfg, pretrained=False).to(cfg.device)
            payload = torch.load(root / 'radial_corrected/fold1_best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(payload['ema'])
            del payload
            report = sol.evaluate_detailed(model, dataset, assess_cfg)
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
    finally:
        sol.get_fold, sol.FilamentDataset = original_fold, OriginalDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
