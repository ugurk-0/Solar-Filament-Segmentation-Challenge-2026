"""Render fixed calibration examples from cached predictions without GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import solution as sol
import pq_experiment as ex
from yolo_data import disjoint_masks
from yolo_dashboard import save_preview, write_json


def main(run):
    run = Path(run)
    state = json.loads((run / 'state.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    checkpoint = run / 'best_pq.pt'
    signature = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if signature != selection['checkpoint_sha256']:
        raise ValueError('Preview checkpoint does not match calibrated predictions')
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network = stored.get('ema') or stored.get('model')
    write_json(run / 'model_identity.json', dict(scale=network.yaml.get('scale'),
               parameters=sum(p.numel() for p in network.parameters()), checkpoint_sha256=signature,
               corrected_label=(run.name == 'yolo11m_20261003')))
    del network, stored
    cfg = ex.config(run)
    coco, _, validation = sol.get_fold(cfg)
    _, dataset = sol.make_loader(coco, validation, cfg, False)
    parameters = state['selected']['parameters']
    cache = run / ('predictions_' + signature[:12])
    for image_id in protocol['split']['calibration'][:2]:
        with np.load(cache / f'{image_id}.npz') as saved:
            keep = saved['scores'] >= parameters['confidence']
            masks = np.unpackbits(saved['masks'][keep], axis=-1, count=cfg.img_size).astype(bool)
            scores = saved['scores'][keep]
        pred = disjoint_masks(masks, scores, dataset.limb, **parameters)
        gt = sol.ground_truth_instances(dataset, image_id)
        row = sol.image_metrics(pred, gt)
        image = sol.load_image(Path(cfg.data_dir) / 'train_images' / coco.imgs[image_id]['file_name'])
        print(save_preview(run, 'selected_calibration', image_id, image, gt, pred, row,
                           epoch=state['best_epoch'], phase='calibration example', parameters=parameters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='+')
    for run in parser.parse_args().run:
        main(run)
