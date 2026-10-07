"""Lightweight notebook views of YOLO artifacts. No torch or model loading."""
import asyncio
import html
import io
import json
import os
from pathlib import Path
import time
import numpy as np

_panel = None


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def write_live(root, phase, **values):
    import psutil
    write_json(Path(root) / 'live.json', dict(phase=phase, updated_at=time.time(),
               pid=os.getpid(), process_started=psutil.Process().create_time(), **values))


def process_active(live, now=None):
    if not live:
        return False
    try:
        import psutil
        process = psutil.Process(live['pid'])
        return process.is_running() and abs(process.create_time() - live['process_started']) < .1
    except (ImportError, KeyError, OSError):
        return False
    except Exception:  # Includes process exit races / psutil.NoSuchProcess.
        return False


def discover_runs(root='runs'):
    return sorted([p.parent for p in Path(root).glob('yolo11*/state.json')],
                  key=lambda p: (p / 'state.json').stat().st_mtime, reverse=True)


def snapshot(run):
    run = Path(run)
    state, protocol, live = [read_json(run / name) for name in ('state.json', 'protocol.json', 'live.json')]
    identity = read_json(run / 'model_identity.json') or state.get('actual_model', protocol.get('actual_model', {}))
    active = state.get('status') not in ('complete', 'failed') and process_active(live)
    status = state.get('status', 'No state recorded')
    if status not in ('complete', 'failed'):
        status = f'Running: {live.get("phase", status)}' if active else f'{status} (no active process verified)'
    scale = identity.get('scale')
    model = f'YOLO11{scale}-seg' if scale else protocol.get('model', 'Model not recorded')
    if identity.get('corrected_label'):
        model += ' — corrected historical label'
    return dict(run=run, state=state, protocol=protocol, live=live, active=active,
                status=status, model=model, previews=read_json(run / 'preview_manifest.json'))


def save_preview(root, tag, image_id, image, gt, pred, metrics, *, epoch=None, phase='monitor', parameters=None):
    """Downsample only the display; metrics and masks are already native-coordinate."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if image.ndim != 2 or any(mask.shape != image.shape for mask in [*gt, *pred]):
        raise ValueError('Preview image and masks must share native coordinates')
    h, w = image.shape
    largest = max(gt or pred, key=lambda mask: int(mask.sum()), default=None)
    if largest is not None and largest.any():
        ys, xs = np.where(largest)
        zoom = (max(0, ys.min()-80), min(h, ys.max()+81), max(0, xs.min()-80), min(w, xs.max()+81))
    else:
        zoom = (h//4, 3*h//4, w//4, 3*w//4)
    figure = Figure(figsize=(13, 8), dpi=110, layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    for row, bounds in enumerate(((0, h, 0, w), zoom)):
        y0, y1, x0, x1 = map(int, bounds)
        stride = max(1, int(np.ceil(max(y1-y0, x1-x0)/600)))
        region = (slice(y0, y1, stride), slice(x0, x1, stride))
        background = image[region]
        for col, (title, masks) in enumerate((('H-alpha image', []), (f'Annotations: {len(gt)}', gt), (f'Predictions: {len(pred)}', pred))):
            axis = axes[row, col]
            axis.imshow(background, cmap='gray', vmin=0, vmax=1)
            if masks:
                labels = np.zeros(background.shape, np.int16)
                for index, mask in enumerate(masks, 1):
                    labels[mask[region]] = (index - 1) % 20 + 1
                axis.imshow(np.ma.masked_equal(labels, 0), cmap='tab20', vmin=1, vmax=20, alpha=.65, interpolation='nearest')
            axis.set_title(('Full disk | ' if row == 0 else 'Detail | ') + title, fontsize=10)
            axis.axis('off')
    figure.suptitle(f'{image_id} | {phase} | epoch {epoch if epoch is not None else "selected"}\n'
                   f'PQ {metrics["pq"]:.3f} | TP {metrics["tp"]} / FP {metrics["fp"]} / FN {metrics["fn"]}', fontsize=12)
    figure.supxlabel('Colors identify instances within each panel. Detail uses the largest annotated instance; PQ uses full-resolution masks.', fontsize=9)
    root = Path(root)
    destination = root / 'previews' / tag / f'{image_id}.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.png')
    figure.savefig(temporary)
    temporary.replace(destination)
    figure.clear()
    manifest = read_json(root / 'preview_manifest.json')
    entries = manifest.setdefault('sets', {})
    group = entries.setdefault(tag, dict(epoch=epoch, phase=phase, parameters=parameters, images=[]))
    group['images'] = [row for row in group['images'] if row['image_id'] != image_id]
    group['images'].append(dict(image_id=image_id, path=destination.relative_to(root).as_posix(), metrics=metrics))
    manifest['latest'] = tag
    write_json(root / 'preview_manifest.json', manifest)
    return destination


def curve_png(info):
    import pandas as pd
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(12, 3.3), dpi=100, layout='constrained')
    FigureCanvasAgg(figure)
    pq_axis, loss_axis = figure.subplots(1, 2)
    history = info['state'].get('history', [])
    if history:
        pq_axis.plot([r['epoch'] for r in history], [r['monitor']['mean_pq'] for r in history], 'o-', label='Monitor mean PQ')
        selected = next((r for r in history if r['epoch'] == info['state'].get('best_epoch')), None)
        if selected:
            pq_axis.scatter([selected['epoch']], [selected['monitor']['mean_pq']], marker='*', s=140, label='Selected checkpoint', zorder=3)
    control = info['state'].get('initial_monitor', {}).get('mean_pq')
    if control is not None:
        pq_axis.axhline(control, color='gray', linestyle='--', label='Initial checkpoint')
    pq_axis.set(title='Monitoring PQ (not assessment PQ)', xlabel='Epoch', ylabel='PQ', ylim=(0, 1))
    if history or control is not None:
        pq_axis.legend(fontsize=8)
    try:
        frame = pd.read_csv(info['run'] / 'training/results.csv')
        frame.columns = frame.columns.str.strip()
        for name in ('train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss'):
            if name in frame:
                loss_axis.plot(frame['epoch'], frame[name], label=name.split('/')[1])
        loss_axis.legend(fontsize=8)
    except (OSError, ValueError):
        loss_axis.text(.5, .5, 'Loss history appears after epoch 1', ha='center', transform=loss_axis.transAxes)
    loss_axis.set(title='Training losses', xlabel='Epoch', ylabel='Loss')
    for axis in (pq_axis, loss_axis):
        axis.grid(alpha=.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png')
    figure.clear()
    return buffer.getvalue()


def display_snapshot(run, preview='latest'):
    from IPython.display import display, HTML, Image
    import pandas as pd
    info = snapshot(run)
    state = info['state']
    display(HTML(f'<h3>{html.escape(info["run"].name)}</h3><b>{html.escape(info["model"])}</b> · {html.escape(info["status"])}'))
    live = info['live']
    if info['active']:
        display(HTML(f'Epoch {live.get("epoch", "?")}/{live.get("epochs", "?")} '
                     f'&middot; last progress update {max(0, int(time.time()-live["updated_at"]))} seconds ago'))
    if info['active'] and live.get('batches'):
        display(HTML(f'Epoch {live.get("epoch", "?")} · batch {live.get("batch", 0)}/{live["batches"]} · '
                     f'GPU reserved {live.get("gpu_gib", 0):.2f} GiB'))
    summaries = {}
    if state.get('initial_monitor'):
        summaries['Initial monitor'] = state['initial_monitor']
    if state.get('history'):
        summaries['Latest monitor'] = state['history'][-1]['monitor']
    if state.get('selected'):
        summaries['Calibration'] = state['selected']['summary']
    if state.get('assessment'):
        summaries['Assessment (85 reused observations)'] = state['assessment']
    if summaries:
        frame = pd.DataFrame(summaries).T
        display(frame[[c for c in ('n', 'mean_pq', 'dataset_pq', 'tp', 'fp', 'fn') if c in frame]].round(4))
    display(HTML(f'Selected epoch: <b>{state.get("best_epoch", "pending")}</b> · Decision: '
                 f'<b>{html.escape(state.get("decision", "pending"))}</b>. Local PQ is not a Kaggle score.'))
    display(Image(data=curve_png(info)))
    manifest = info['previews']
    key = manifest.get('latest') if preview == 'latest' else preview
    group = manifest.get('sets', {}).get(key, {})
    if not group:
        display(HTML('Prediction previews appear after the first monitoring pass.'))
    else:
        display(HTML(f'<b>Images: {html.escape(str(key))}</b> · {html.escape(str(group.get("parameters", {})))}'))
        for record in group.get('images', [])[:2]:
            path = info['run'] / record['path']
            if path.exists():
                display(Image(filename=str(path)))
    return info


def dashboard(root='runs', auto_refresh=True, interval=15):
    """Run selector + preview selector + nonblocking automatic refresh."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    global _panel
    if _panel is not None:
        _panel.close()
    selector = widgets.Dropdown(description='Run:', layout=widgets.Layout(width='650px'))
    previews = widgets.Dropdown(description='Images:', options=['latest'], layout=widgets.Layout(width='350px'))
    auto = widgets.ToggleButton(value=auto_refresh, description='Auto refresh', icon='refresh')
    refresh = widgets.Button(description='Refresh now')
    output = widgets.Output()

    class Panel:
        task = None
        updating = False

        def refresh(self, *_):
            if self.updating:
                return
            self.updating = True
            try:
                runs = discover_runs(root)
                selected = selector.value
                options = [('Auto: active or latest run', 'auto')] + [(p.name, str(p)) for p in runs]
                selector.options = options
                selector.value = selected if selected in [v for _, v in options] else 'auto'
                run = next((p for p in runs if snapshot(p)['active']), runs[0] if runs else None) if selector.value == 'auto' else Path(selector.value)
                if run:
                    manifest = read_json(run / 'preview_manifest.json')
                    old_preview = previews.value
                    previews.options = ['latest'] + list(reversed(manifest.get('sets', {})))
                    previews.value = old_preview if old_preview in previews.options else 'latest'
                with output:
                    clear_output(wait=True)
                    if run:
                        display_snapshot(run, previews.value)
                    else:
                        print('No local YOLO run artifacts found under', Path(root).resolve())
            finally:
                self.updating = False

        def close(self):
            if self.task is not None:
                self.task.cancel()
            auto.value = False

        async def watch(self):
            while True:
                await asyncio.sleep(interval)
                if auto.value:
                    self.refresh()

    panel = Panel()
    selector.observe(panel.refresh, names='value')
    previews.observe(panel.refresh, names='value')
    refresh.on_click(panel.refresh)
    display(widgets.VBox([selector, widgets.HBox([previews, auto, refresh]), output]))
    panel.refresh()
    try:
        panel.task = asyncio.get_running_loop().create_task(panel.watch())
    except RuntimeError:
        auto.value = False
        auto.disabled = True
    _panel = panel
    return panel
