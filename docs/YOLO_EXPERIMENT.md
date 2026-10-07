# YOLO instance-segmentation experiment

The hypothesis is that a detector with a separate mask per object can recover
filament identity more reliably than dividing a single foreground map into
connected components. This is a test of **YOLO11n-seg**, not a claim that every
YOLO architecture will behave the same way. Its measured results and training
history are in the [experiment report](../reports/yolo11n_20261003.md).

There is task-specific precedent: [Diercke et al.](https://arxiv.org/abs/2402.15407)
combine object detection and U-Net segmentation for solar H-alpha observations.
The public [HeShen competition implementation](https://github.com/HeShen-1/filament-segmentation-2026)
uses YOLO11m-seg at 1536 pixels followed by a seed-conditioned crop refiner.
That author reports 0.3593 five-fold mean PQ and 0.35 public leaderboard PQ for
their respective systems. These are external results with different models,
splits, and fusion settings; they are not measurements of this repository.
Our smaller direct-mask experiment first tests whether useful instance
proposals can be learned within the local compute budget. A learned crop
refiner remains a follow-up hypothesis. The cheap probability-map refinement
below reuses our existing model and is not a reproduction of their refiner.

## Model and efficiency choices

The starting point is the official COCO-pretrained `yolo11n-seg.pt`. The nano
model leaves room for 1536-pixel inputs on the local 6 GB RTX 3060 Laptop GPU.
We prioritize image resolution because many filaments are narrow. The larger
models suggested in the previous handoff remain untested here.

Training uses mixed precision, batch 2, nominal batch 8 with gradient
accumulation, AdamW, and a cosine learning-rate schedule. The predeclared
budget is at most 20 epochs, with PQ-based early stopping after at least 8
epochs and 6 epochs without a monitoring improvement greater than 0.001.
Worker count is zero to avoid Windows worker startup and memory overhead.
Prepared PNGs and converted labels are cached on disk.
An initial batch-1 capacity probe used about 2.8 GB of the 6 GB GPU at 1536
pixels. It was stopped before completing its first epoch, archived, and
restarted from the same pretrained weights with batch 2 to use that headroom.
It is not counted as a completed training experiment or a PQ result.

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
An initial preparation pass found an instance with internal holes: extracting
only its outer boundary produced IoU 0.972249 and correctly stopped the run.
The converter now includes hole boundaries and retraces each connecting bridge
instead of chaining holes across the object. Both observed failure cases
round-trip exactly (IoU 1.0); synthetic single-hole and concave multi-hole
regression tests guard this behavior.

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

## Optional refinement without more training

`experiment_yolo_refinement.py` tests whether the retained U-Net can repair
YOLO mask boundaries. For each YOLO instance, dilate its mask locally, intersect
that neighborhood with the U-Net foreground probability threshold, apply the
solar support, then assign overlapping pixels by YOLO confidence. It preserves
the YOLO object's identity and cannot invent a new detection far from a proposal.

The grid was specified during YOLO training, before its assessment: dilation
radii 8/24 native pixels, U-Net thresholds 0.5/0.8, YOLO confidences 0.1/0.3,
minimum area 128. These eight settings reuse cached full-resolution predictions.
Only a setting with calibration mean PQ more than 0.005 above calibrated direct
YOLO **and** higher pooled PQ advances to assessment. The final promotion gate
compares against the retained direct YOLO candidate. This
limits extra compute and avoids reporting a calibration-only gain as an
assessment improvement.

```powershell
python experiment_yolo_refinement.py --smoke
python experiment_yolo_refinement.py
```

Run these after the corresponding direct YOLO experiment finishes. Outputs go
under `runs/yolo_refinement_20261006/` and the tracked report records the decision.
The completed trial was rejected: best calibration mean PQ 0.2865 versus
0.2854, but pooled PQ fell from 0.2889 to 0.2804. No assessment was performed.

## Continuing training from the retained checkpoint

The completed nano model reached **0.3199 mean / 0.3319 pooled assessment PQ**.
The October 6 continuation uses `train_yolo_experiment.py`, a configurable
runner that verifies the actual architecture before training. The earlier
file named `experiment_yolo11m.py` accidentally loaded nano weights; its report
is corrected and the weight-path bug is fixed. It provides no medium-model
result. See the [audit](../reports/yolo_audit_20261006.md).

```powershell
python train_yolo_experiment.py --variant n --weights runs/yolo11n_20261003/best_pq.pt --output runs/yolo11n_finetune_20261006 --epochs 12 --lr0 0.00015 --batch 2 --smoke
python train_yolo_experiment.py --variant n --weights runs/yolo11n_20261003/best_pq.pt --output runs/yolo11n_finetune_20261006 --epochs 12 --lr0 0.00015 --batch 2
```

This is fine-tuning with a fresh optimizer and cosine schedule, not resuming
the exact old optimizer state. It retains the parent as epoch-zero control,
reuses prepared data, and stops after at least six epochs if five epochs fail
to improve monitoring PQ by more than 0.001. Prediction uses a separate model
instance so inference-time Conv/BN fusion cannot alter training initialization.
The minimum inference confidence equals the lowest threshold needed by the
current evaluation, avoiding reconstruction of masks that would be discarded.
The full-resolution parent control reproduced monitoring PQ **0.3417497475**
exactly after this optimization. The [live report](../reports/yolo11n_finetune_20261006.md)
records the new run; monitoring values are distinct from assessment scores.

For a verified model-size experiment, omit `--weights` so the runner downloads
the requested official pretrained variant and checks its architecture. The
prepared observation dataset is shared across runs; checkpoints and reports
remain isolated. The next small-model trial is configured as:

```powershell
python train_yolo_experiment.py --variant s --output runs/yolo11s_20261006 --epochs 24 --lr0 0.001 --batch 1 --smoke
python train_yolo_experiment.py --variant s --output runs/yolo11s_20261006 --epochs 24 --lr0 0.001 --batch 1 --update-readme
```

`--update-readme` refreshes the result row from the final recorded state when
training/evaluation ends. It only promotes a headline result after the paired
comparison gate passes. If a stronger candidate has been retained before a
new run, pass its run directory with `--baseline`.

## Reproduce

### Watch runs in the notebook

Install `requirements-notebook.txt` in the selected notebook kernel, then execute
the two cells under **YOLO runs and prediction images** in `notebook.ipynb`.
The static view includes the retained nano model's local curves and actual cached
calibration predictions. The widget selects the active run automatically and
refreshes every 15 seconds; a dropdown selects a different run or preview epoch.

`train_yolo_experiment.py` writes `live.json` with process identity and batch progress,
and its existing monitoring pass saves two fixed observations per epoch. No second
prediction pass is needed. `preview_manifest.json` records the image IDs, epoch,
thresholds and native-resolution per-image metrics. PNGs show full-disk and detail
views of the image, annotations and predicted instances. These examples illustrate
behavior and do not replace the aggregate assessment.

The display module, `yolo_dashboard.py`, loads only status, CSV and PNG artifacts;
it does not import Torch or load another model onto the GPU. Its process check
avoids calling an abandoned run active. Stop **Auto refresh** to pause the view;
this does not stop the separate training process. Re-executing the cell replaces
the earlier refresh task.

For completed historical runs, rebuild the fixed calibration examples from their
checkpoint-verified prediction caches (no GPU inference):

```powershell
python -m scripts.build_yolo_previews runs/yolo11n_20261003 runs/yolo11n_finetune_20261006
```

Use the YOLO environment for this cache conversion. The dashboard itself uses
the notebook environment. Run only one training/evaluation GPU job at a time on
the 6 GB development GPU. Local images remain under `runs/`; Git strips notebook
outputs and does not publish trained weights or prediction caches.

### Environment and commands

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
