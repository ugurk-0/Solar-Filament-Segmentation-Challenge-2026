# YOLO instance-segmentation experiment

The hypothesis is that a detector with a separate mask per object can recover
filament identity more reliably than dividing a single foreground map into
connected components. This is a test of **YOLO11n-seg**, not a claim that every
YOLO architecture will behave the same way. Its measured results and training
history are in the [experiment report](../reports/yolo11n_20261003.md).

## Model and efficiency choices

The starting point is the official COCO-pretrained `yolo11n-seg.pt`. The nano
model leaves room for 1536-pixel inputs on the local 6 GB RTX 3060 Laptop GPU.
We prioritize image resolution because many filaments are narrow. The larger
models suggested in the previous handoff remain untested here.

Training uses mixed precision, batch 1, nominal batch 8 with gradient
accumulation, AdamW, and a cosine learning-rate schedule. The predeclared
budget is at most 20 epochs, with PQ-based early stopping after at least 8
epochs and 6 epochs without a monitoring improvement greater than 0.001.
Worker count is zero to avoid Windows worker startup and memory overhead.
Prepared PNGs and converted labels are cached on disk.

The head produces low-resolution mask prototypes and per-detection mask
coefficients. Their linear combination is resized to the original 2048-pixel
image, cropped to the predicted box, and thresholded. The 1536-pixel input
gives a 384-pixel prototype grid in this pinned model. `mask_ratio=4` sets the
training target raster resolution; changing that option alone does not
increase the model's prototype resolution. `retina_masks=True` preserves
native output coordinates, but cannot restore detail the model never learned.

Native mask reconstruction originally exhausted GPU memory in the smoke test.
The predictor now calls the upstream reconstruction in chunks of eight
detections and moves completed masks to CPU. A regression test checks exact
mask and box equality against the upstream implementation across chunk
boundaries. It retains all detections up to the declared 100-instance limit.

## Data and solar annotation support

The fold comes from the existing physical-observation split. Each of the 566
training observations contributes its first actual annotator record, so
repeated annotation records do not increase an observation's sampling weight.
This selection does not synthesize a consensus label. Validation keeps the
existing deduplicated annotation choice and the 16/40/85 monitoring,
calibration, and assessment partition.

Images are grayscale, replicated to three channels by the YOLO loader. Pixels
outside the existing solar annotation-support mask are zeroed. All chirality
categories become the single detection class `filament`; each polygon remains
a separate object. Polygon conversion checks the mask round-trip IoU and
fails below 0.98. The preparation manifest records the observed minimum and
mean, including any contour bridges needed for disconnected pieces. Evaluation
uses the original rasterized annotations, not the converted YOLO labels.

Zeroing the image alone would still teach the classifier from unannotated
regions. The custom loss therefore excludes unsupported classification anchors,
removes their positive assignments, and computes mask loss only inside the
annotation support and object box. Geometric augmentation, mosaic, mixup, and
multi-scale resizing are disabled so this support remains aligned. Brightness
augmentation is enabled. The regression suite checks zero mask-loss gradients
outside the annotation region.

## PQ selection and interpretation

Ultralytics also logs detection and mask mAP. Checkpoint selection instead uses
the repository's PQ implementation on the 16 monitoring observations, at
confidence 0.1 and minimum instance area 128 native pixels. Confidence-ordered
ownership makes overlapping predicted masks disjoint while preserving object
identity. The annotation-support mask applies again at inference.

After training, confidence thresholds 0.05/0.1/0.2/0.3/0.5 and minimum areas
32/128/500 are compared only on the 40 calibration observations. Predictions
are cached once per image and checkpoint. The frozen selection is evaluated
on the 85 assessment observations and compared with the skeleton-merge
candidate using paired bootstrapping. Promotion requires higher pooled PQ and
a positive lower confidence bound for the mean-PQ difference.

The assessment set has been reused across experiments. Its score is exploratory
local evidence, not an unbiased final test or a Kaggle leaderboard result.
This runner does not submit anything to Kaggle or replace a submission CSV.

## Reproduce

Use a separate Python 3.12 environment so the optional YOLO dependencies do not
alter the established U-Net environment. Install a CUDA-enabled PyTorch build
compatible with the pinned requirements and local GPU before training.

```powershell
python -m venv .venv-yolo
& ./.venv-yolo/Scripts/python.exe -m pip install -r requirements-yolo.txt
& ./.venv-yolo/Scripts/python.exe -m pytest tests -q
& ./.venv-yolo/Scripts/python.exe experiment_yolo.py --smoke
& ./.venv-yolo/Scripts/python.exe experiment_yolo.py
```

On the development machine, the separate interpreter is
`C:/Users/ugurk/AppData/Local/FilamentYolo/venv/Scripts/python.exe` and inherits
the existing CUDA PyTorch installation. The script expects competition data
at the repository's existing local data path and the skeleton-merge assessment
artifact for the paired comparison. These data and checkpoints are not bundled.

Outputs are under `runs/yolo11n_20261003/`: source/split protocol, dataset
manifest, per-epoch state, `best_pq.pt`, calibration selection, and assessment.
The report is tracked by Git; data and checkpoints are ignored. Completed runs
are preserved, and changed source/data inputs require a new output directory.
An interrupted training run is retained rather than silently restarted.

## Primary references

- [Ultralytics YOLO11 models](https://docs.ultralytics.com/models/yolo11): pretrained segmentation variants.
- [Instance segmentation](https://docs.ultralytics.com/tasks/segment): per-instance mask outputs and training interface.
- [Training options](https://docs.ultralytics.com/modes/train): resolution, accumulation, augmentation, and mask settings.
- [Callbacks](https://docs.ultralytics.com/usage/callbacks): epoch-end custom evaluation.
- [Pinned segmentation predictor source](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/models/yolo/segment/predict.py): native-mask reconstruction.

The optional Ultralytics dependency is distributed under AGPL-3.0, with an
enterprise licensing option described in its documentation.
