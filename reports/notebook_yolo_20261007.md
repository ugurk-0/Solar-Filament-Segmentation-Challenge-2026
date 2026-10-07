# YOLO notebook visibility — 7 October 2026

Hypothesis: fixed per-epoch examples alongside monitoring PQ and loss curves make
missed, merged and fragmented filaments visible without running a second model in
the notebook or adding a second inference pass to training.

## Implementation

- `yolo_dashboard.py`: process-verified status, 15-second interactive refresh,
  run/epoch selectors, PQ and loss curves, and full-disk/detail instance overlays.
- `experiment_yolo.py` and `train_yolo_experiment.py`: an optional observer reuses
  existing monitoring predictions; two fixed observations are rendered per epoch.
  Batch heartbeat records PID, process creation time, epoch, batch and GPU reserve.
- `scripts/build_yolo_previews.py`: historical previews from the selected
  checkpoint's hash-verified calibration cache. Original nano, fine-tuned nano,
  and the historically mislabeled nano run now have actual examples locally.
- `build_notebook.py` and `notebook.ipynb`: a self-contained YOLO section before
  earlier U-Net experiments. Notebook-aware updates preserve other cells and
  outputs; the prior notebook is backed up in `runs/notebook_ui_20261007/`.
- `requirements-notebook.txt`: optional widget dependency pinned to 8.1.5.
  `scripts/validate_yolo_notebook.py` validates the cells in an existing kernel.

## Validation and observed results

`python -m pytest tests/test_yolo_dashboard.py -q`: **2 passed**. Checks native
coordinates, actual PNG creation, manifest replacement, partial JSON handling,
PID reuse protection, completed-run status and curve generation.

`python -m pytest tests -q`: **69 passed** in 11.65 seconds in the separate YOLO
Python 3.12 environment. No training job ran concurrently with tests.

`python -m scripts.build_yolo_previews runs/yolo11n_20261003
runs/yolo11n_finetune_20261006 runs/yolo11m_20261003`: six real preview images
generated. A full-size PNG was visually inspected for readable panels, coordinate
alignment and visible thin masks. Colors are independent in each panel.

`python -m scripts.validate_yolo_notebook <selected-kernel-connection-file>`:
validated imports from the repository root in the existing VS Code Python 3.12
kernel, a table, one curve image, two prediction images, widget construction,
the asynchronous task and manual refresh callback. Static outputs were saved
locally; the interactive cell is run to create its live display. The validation
task was closed afterward. Connection credentials are not recorded here.

## Score interpretation

Best retained local mean PQ remains **0.3199**, pooled **0.3319**, on the reused
85-observation assessment. Fine-tuning gave mean **0.3173**, pooled **0.3345** and
was not promoted. Monitoring curves and individual-image PQ are not aggregate
assessment or leaderboard results. No new Kaggle score is claimed.

## Training execution check

`python -u train_yolo_experiment.py --variant s --output runs/yolo11s_20261006
--epochs 24 --lr0 0.001 --batch 1 --smoke`: completed training, monitoring,
calibration and assessment. The smoke uses four training observations, 640-pixel
input and one epoch; its two-observation assessment PQ was 0.0, an execution
check with no performance conclusion. CSV losses were finite. The trained
architecture was verified as **YOLO11s-seg, 10,082,675 parameters**. Both
`epoch_000` and `epoch_001` contain two real prediction previews.

The same command without `--smoke`, with `--update-readme`, was launched as a
separate background process at full 1536-pixel resolution, batch 1, and a budget
of 24 epochs with the existing early-stopping rule. Its running state, preview
images and eventual assessment are recorded in `runs/yolo11s_20261006/` and
`reports/yolo11s_20261006.md`. This report records launch-time validation; consult
the notebook's process-verified status for whether training is still active.

First full-resolution epoch verified: monitoring mean PQ **0.1611** on 16 observations, finite recorded losses, and two epoch-1 prediction images. Epoch 2 was active at this check. This early monitoring value is not the final calibrated assessment.
