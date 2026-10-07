import os
import numpy as np
import pytest
from yolo_dashboard import (save_preview, read_json, write_json, write_live,
                            process_active, snapshot, discover_runs, curve_png)


def test_preview_native_coordinates_and_manifest(tmp_path):
    image = np.zeros((100, 120), np.float32)
    mask = image.astype(bool)
    mask[20:70, 40:45] = True
    metrics = dict(pq=1., tp=1, fp=0, fn=0)
    path = save_preview(tmp_path, 'epoch_001', 7, image, [mask], [mask], metrics, epoch=1)
    assert path.read_bytes().startswith(b'\x89PNG')
    manifest = read_json(tmp_path / 'preview_manifest.json')
    assert manifest['latest'] == 'epoch_001'
    assert manifest['sets']['epoch_001']['images'][0]['metrics']['pq'] == 1.
    save_preview(tmp_path, 'epoch_001', 7, image, [], [], metrics, epoch=1)
    assert len(read_json(tmp_path / 'preview_manifest.json')['sets']['epoch_001']['images']) == 1
    with pytest.raises(ValueError, match='native coordinates'):
        save_preview(tmp_path, 'bad', 7, image, [mask[:3]], [], metrics)


def test_live_status_is_process_verified(tmp_path):
    run = tmp_path / 'yolo11s_example'
    write_json(run / 'state.json', dict(status='training', history=[]))
    write_live(run, 'training', batch=4, batches=20)
    info = snapshot(run)
    assert info['active'] and info['live']['pid'] == os.getpid()
    assert discover_runs(tmp_path) == [run]
    assert curve_png(info).startswith(b'\x89PNG')
    assert not process_active(dict(info['live'], process_started=0))
    write_json(run / 'state.json', dict(status='complete'))
    assert not snapshot(run)['active']
    (run / 'state.json').write_text('{')
    assert read_json(run / 'state.json') == {}
