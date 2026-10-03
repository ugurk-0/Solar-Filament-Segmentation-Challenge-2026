"""Ultralytics extensions that exclude unannotated solar regions from losses.

Written against ultralytics 8.3.253; no upstream implementation is copied.
Fixed square geometry is required. Photometric augmentation remains available.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.models.yolo.segment.train import SegmentationTrainer
import solution as sol


@lru_cache(maxsize=12)
def support_tensor(size, device):
    cfg = sol.Cfg(img_size=size, disk_radius=950 * size / 2048)
    return torch.from_numpy(sol.build_limb_mask(cfg)).to(device).float()


class SupportBCE(torch.nn.BCEWithLogitsLoss):
    def forward(self, prediction, target):
        return super().forward(prediction, target) * self.valid


class SupportAssigner(torch.nn.Module):
    def __init__(self, assigner):
        super().__init__()
        self.inner = assigner

    def forward(self, *args, **kwargs):
        labels, boxes, scores, foreground, indices = self.inner(*args, **kwargs)
        valid = self.valid.squeeze(-1).bool()
        return labels, boxes, scores * self.valid, foreground & valid, indices


class SupportSegmentationLoss(v8SegmentationLoss):
    def __init__(self, model):
        super().__init__(model)
        for name in ('mosaic', 'mixup', 'copy_paste', 'degrees', 'translate', 'scale', 'shear',
                     'perspective', 'flipud', 'fliplr', 'multi_scale'):
            if getattr(model.args, name, 0):
                raise ValueError(f'Support-aware loss requires fixed geometry: {name}=0')
        self.bce = SupportBCE(reduction='none')
        self.assigner = SupportAssigner(self.assigner)

    def __call__(self, predictions, batch):
        feats = predictions[0] if len(predictions) == 3 else predictions[1][0]
        image = batch['img']
        if image.shape[-2] != image.shape[-1]:
            raise ValueError('Support-aware loss expects square images')
        support = support_tensor(image.shape[-1], image.device)[None, None]
        valid = torch.cat([F.interpolate(support, feature.shape[-2:], mode='nearest').flatten(2)
                           for feature in feats], dim=2).transpose(1, 2)
        self.bce.valid = self.assigner.valid = valid
        return super().__call__(predictions, batch)

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        logits = torch.einsum('in,nhw->ihw', pred, proto)
        support = support_tensor(proto.shape[-1], proto.device)[None].expand(len(gt_mask), -1, -1).clone()
        support = crop_mask(support, xyxy)
        error = F.binary_cross_entropy_with_logits(logits, gt_mask, reduction='none')
        return ((error * support).sum((1, 2)) / support.sum((1, 2)).clamp_min(1)).sum()


class SupportSegmentationModel(SegmentationModel):
    def init_criterion(self):
        return SupportSegmentationLoss(self)


class SupportSegmentationTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SupportSegmentationModel(cfg, ch=self.data['channels'], nc=self.data['nc'], verbose=verbose)
        if weights:
            model.load(weights)
        return model
