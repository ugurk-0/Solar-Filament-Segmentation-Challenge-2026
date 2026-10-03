"""Bound native-resolution mask reconstruction memory without changing masks."""
import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor


class ChunkedSegmentationPredictor(SegmentationPredictor):
    def construct_result(self, pred, img, orig_img, img_path, proto):
        if not len(pred):
            return super().construct_result(pred, img, orig_img, img_path, proto).cpu()
        parts = [super(ChunkedSegmentationPredictor, self).construct_result(
            chunk.clone(), img, orig_img, img_path, proto).cpu()
            for chunk in pred.split(8)]
        return Results(orig_img, path=img_path, names=self.model.names,
                       boxes=torch.cat([part.boxes.data for part in parts]),
                       masks=torch.cat([part.masks.data for part in parts]))
