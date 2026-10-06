"""Refresh the narrow YOLO result rows from completed, recorded experiments."""
import json
from pathlib import Path


RUNS = {
    'yolo11n_finetune_20261006': 'YOLO11n lower-rate fine-tuning',
    'yolo11s_20261006': 'YOLO11s, batch 1, 24-epoch budget',
}


def update_readme(repo=Path('.')):
    repo = Path(repo)
    readme = repo / 'README.md'
    content = readme.read_text(encoding='utf-8')
    for name, label in RUNS.items():
        path = repo / 'runs' / name / 'state.json'
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state['status'] not in ('complete', 'failed'):
            continue
        measured = state.get('assessment')
        mean, pooled = ((f"{measured['mean_pq']:.4f}", f"{measured['dataset_pq']:.4f}")
                        if measured else ('—', '—'))
        decision = {'no_monitor_gain': 'Rejected on monitoring; parent retained',
                    'promoted_research_candidate': 'Promoted research candidate',
                    'not_promoted': 'Did not pass paired assessment gate'}.get(
                        state.get('decision'), state['status'])
        row = f'| {label} | {mean} | {pooled} | {decision}; [report](reports/{name}.md) |'
        lines = content.splitlines()
        existing = next((i for i, line in enumerate(lines) if line.startswith(f'| {label} |')), None)
        if existing is not None:
            lines[existing] = row
        else:
            anchor = next((i for i, line in enumerate(lines) if line.startswith('| YOLO instances + retained U-Net boundaries |')), None)
            if anchor is None:
                continue
            lines.insert(anchor, row)
        content = '\n'.join(lines) + '\n'
    # A run must pass its comparison gate to become a headline candidate.
    candidates = []
    for name in ('yolo11n_20261003', *RUNS):
        path = repo / 'runs' / name / 'state.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('status') == 'complete' and state.get('promoted') and state.get('assessment'):
                candidates.append((name, state))
    if candidates:
        name, state = max(candidates, key=lambda entry: entry[1]['assessment']['mean_pq'])
        measured = state['assessment']
        label = 'YOLO11n-seg' if name == 'yolo11n_20261003' else RUNS[name]
        start = content.index('**Best retained research result:**')
        end = content.index('\n\n', start)
        content = (content[:start] + f"**Best retained research result:** **{measured['mean_pq']:.4f} mean PQ / "
                   f"{measured['dataset_pq']:.4f} pooled PQ**, using {label} "
                   f"(epoch {state['best_epoch']} selected by monitoring PQ). "
                   'This is an exploratory comparison on 85 reused local observations, not a leaderboard score. '
                   'The last user-reported Kaggle score remains **0.24** for an earlier submission. '
                   f'See the [measured result](reports/{name}.md) and [October 6 audit](reports/yolo_audit_20261006.md).' + content[end:])
    if content != readme.read_text(encoding='utf-8'):
        readme.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    update_readme()
