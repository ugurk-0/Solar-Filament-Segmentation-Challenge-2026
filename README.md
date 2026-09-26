# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha images from MAGFiLO v1.0, developed for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

The pipeline trains a U-Net, separates predicted filaments into instances, evaluates **Panoptic Quality (PQ)**, and exports an RLE submission CSV. The notebook presents a paired experiment on post-processing and training changes.

**Status:** work in progress. Source code, a presentation notebook, regression tests, and a draft technical report are available. Datasets, trained checkpoints, and generated experiment outputs are not included. No final leaderboard score or completed ablation study is reported here.

## Repository contents

| File | Purpose |
|---|---|
| [solution.py](solution.py) | Training, evaluation, post-processing tuning, and submission CLI |
| [notebook.ipynb](notebook.ipynb) | Presentation and optional execution of the paired PQ experiment |
| [pq_experiment.py](pq_experiment.py) | Calibration, paired checkpoint comparison, and uncertainty estimates |
| [audit_data.py](audit_data.py) | Training-data annotation-support audit |
| [build_notebook.py](build_notebook.py) | Notebook generator using `nbformat` |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |
| [tests/test_solution_regressions.py](tests/test_solution_regressions.py) | Metric, mask, TTA, inference, and training regression checks |
| [main.tex](main.tex) | Draft technical report; quantitative conclusions remain provisional |
| [.github/workflows/tests.yml](.github/workflows/tests.yml) | Regression tests on pushes and pull requests |

## Setup and data

Use Python 3.12, matching the CI configuration. From a terminal:

```bash
git clone https://github.com/ugurk-0/Solar-Filament-Segmentation-Challenge-2026.git
cd Solar-Filament-Segmentation-Challenge-2026
python -m pip install -r requirements.txt
```

CUDA runs require a compatible PyTorch installation. The local development environment used PyTorch 2.4.1 and torchvision 0.19.1 with CUDA 12.4 wheels. CPU execution is supported, but full-resolution validation can be slow.

Download the competition data separately and retain its annotation and image filenames. The examples below assume this layout:

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── train_images/
└── test/
    └── test_images/
```

The CLI defaults to Kaggle paths. For local runs, pass `--data-dir`, `--test-dir`, and `--work-dir` as shown below. Data and generated files under `runs/` are ignored by Git.

## Train, evaluate, and submit

### 1. Smoke test

Use a new output directory so that existing checkpoints remain intact:

```bash
python solution.py train --fold 0 --epochs 2 --limit 8 --no-tta --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/smoke_fold0
python solution.py eval --fold 0 --limit 8 --no-tta --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/smoke_fold0
```

`--limit` caps training and validation images and limits training to at most two epochs. These runs check execution; their scores are not model-quality estimates. Submission rejects checkpoints trained with a nonzero `limit`.

### 2. Train a fold and evaluate it

```bash
python solution.py train --fold 0 --epochs 60 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/fold0
python solution.py eval --fold 0 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/fold0
```

Training includes periodic validation and writes checkpoints, history, and previews. It does not automatically run the separate `tune` or `submit` commands. Use a separate work directory for each additional fold (1–4).

Checkpoint selection matters:

- `fold0.pt` contains the latest epoch, including EMA weights.
- `fold0_best.pt` contains the epoch with the highest monitored mean image PQ.
- `eval` prefers the best checkpoint when it exists, otherwise the latest checkpoint.
- `tune` currently uses the latest checkpoint. `submit` uses the checkpoints explicitly supplied.

Keep checkpoint choice, observation split, annotation support, TTA, and post-processing consistent when comparing experiments. `--eval-max-images` selects a monitoring subset during training; it does not constitute a final full-fold assessment.

### 3. Tune and generate a submission

This single-fold example tunes and submits the same latest checkpoint. It uses the same work directory so submission can load the resulting `best_postproc.json`:

```bash
python solution.py tune --fold 0 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/fold0
python solution.py submit --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images --work-dir runs/fold0 --ckpts runs/fold0/fold0.pt
```

Submission writes `submission.csv` and `submission_manifest.json`. The CSV contains `filament_id` and `segmentation_rle`, with one row per predicted instance. Each encoded mask is checked with an RLE round trip.

To ensemble folds, pass multiple compatible checkpoints after `--ckpts`. Submission looks for tuned settings only in its own `--work-dir`; a new output directory needs the intended `best_postproc.json` copied into it. Otherwise configuration defaults apply. Validate the chosen ensemble and post-processing on held-out observations; do not tune on test images.

## Notebook and paired experiment

Open [notebook.ipynb](notebook.ipynb) in a notebook frontend with the installed Python environment as its kernel, and execute cells in order from the repository root. It imports `solution.py` and `pq_experiment.py` rather than maintaining a second implementation.

All expensive stages are disabled by default:

| Switch | Action |
|---|---|
| `RUN_TRAINING` | Warm-start the configured same-fold checkpoint with a fresh optimizer |
| `RUN_CALIBRATION` | Select post-processing settings for the baseline checkpoint |
| `RUN_ASSESSMENT` | Compare the selected training checkpoint against the baseline |
| `RUN_SUBMISSION` | Generate predictions using the selected checkpoint and settings |

The notebook expects local artifacts under `runs/pq_research_20260926` and a baseline checkpoint at `runs/corrected_loss_fold1/fold1.pt`. These are not distributed. Configure paths and provide a compatible baseline before enabling stages; saved tables and previews appear only when their files exist. For training from scratch, use the CLI workflow above.

The experiment separates fold-1 observations into a fixed monitoring subset, a calibration subset, and the remaining paired assessment subset. Its hypothesis is that connected components reduce watershed fragmentation and that corrected crop sampling, clDice, and annotation support improve training. These combined changes are not an ablation of individual contributions.

The experiment reports paired differences with observation and date-block bootstrap intervals. Full-fold metrics are descriptive because the fold also contains observations used for selection. Historical scores computed with different annotation support or TTA settings are not directly comparable.

## Method and metric

| Component | Current implementation |
|---|---|
| Split | Five-fold GroupKFold using the physical observation key `Path(file_name).stem.split("-")[-1]`; validation deduplicates annotator records |
| Annotation support | Projected ±70° longitude mask with fixed disk radius and zero solar-tilt approximation; applied to losses, predictions, and validation labels |
| Model | ResNet-18 timm encoder, U-Net decoder, attention gates, and residual dilated bottleneck |
| Targets and loss | Segmentation masks; weighted BCE, soft-Dice, and soft-clDice; auxiliary spine loss disabled by default |
| Training | Filament-centred crops, geometric and photometric augmentation, gradient accumulation, AMP on CUDA, cosine scheduling, and EMA |
| Inference | Sliding windows with overlap averaging; eight-way dihedral TTA by default; optional checkpoint ensemble |
| Instance separation | Connected components by default; optional watershed via `--instance-method watershed` |

PQ uses greedy one-to-one instance matching at IoU ≥ 0.5:

```text
SQ = sum(matched IoUs) / TP
RQ = TP / (TP + FP/2 + FN/2)
PQ = SQ × RQ
```

The implementation returns zero when no instances match, including empty prediction/ground-truth pairs. Mean per-image PQ and pooled dataset PQ are different summaries and should be labeled separately.

## Validation and results

```bash
python -m pytest tests/test_solution_regressions.py -q
```

The publication check passed **12 regression tests**. The notebook passed format validation and contains no saved outputs. Those checks do not establish training quality or a competition score.

`eval` writes `evaluation_foldF.json` with aggregate and per-image results: PQ/SQ/RQ, pixel and matched-instance overlap metrics, TP/FP/FN, fragmentation and merging counts, and timing. The paired experiment additionally writes calibration and comparison reports under its configured output directory.

Publish measured scores together with the checkpoint, split, support mask, TTA, and post-processing configuration. The draft [technical report](main.tex) still requires completed experiments before its results can be finalized.
