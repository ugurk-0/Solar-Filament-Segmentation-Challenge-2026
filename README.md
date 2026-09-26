# Solar Filament Segmentation Challenge 2026

Used Codex GPT 6 Astra to assist me in this work.

Instance segmentation of solar filaments in GONG H-alpha images from MAGFiLO v1.0, developed for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

The pipeline trains a U-Net, separates predicted filaments into instances, evaluates **Panoptic Quality (PQ)**, and exports an RLE submission CSV. The notebook presents a paired experiment on post-processing and training changes.

**Status:** one fold-1 post-processing calibration, one four-epoch warm-start run, an independent paired assessment, and three three-epoch PQ-guided follow-ups are complete locally. The selected local checkpoint is still the first epoch of the four-epoch run. Test inference/submission and multi-fold confirmation remain pending. No leaderboard score or completed ablation study is reported here. See the [dated experiment report](reports/pq_experiment_20260926.md) for measured results and limitations.

## Repository contents

For the folder map, see [repository organization](docs/REPOSITORY.md).
For Kaggle, follow the [upload guide](kaggle/README.md). To produce the portable
notebook after installing dependencies:

```bash
python scripts/build_kaggle.py
```

Upload `dist/kaggle/filament-kaggle.ipynb`. It contains the current pipeline source
and configurable training, evaluation, tuning, and submission stages.
The local Windows checkout also supports [automatic GitHub sync](docs/AUTO_SYNC.md).

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
| [reports/pq_experiment_20260926.md](reports/pq_experiment_20260926.md) | Completed local experiment, stage status, metrics, commands, and checkpoint decision |
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
python -m pytest tests -q
```

The publication check passed **15 tests**, including portable Kaggle export and notebook-output filtering. The research notebook was executed locally with saved stage tables, learning curves, and prediction overlays; Git strips those outputs from the published copy while preserving them on disk. The generated Kaggle notebook's setup cells were executed locally with expensive stages disabled; it has not yet been run on Kaggle. These checks and local measurements do not establish a competition score.

### Local fold-1 experiment completed 26 September 2026

The calibration split selected connected components with probability threshold `0.65` and minimum area `200`. On the separate 85-observation assessment subset:

| Checkpoint and inference | Mean image PQ | Dataset PQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Original checkpoint, watershed 0.50/50 | 0.0706 | 0.0706 | 172 | 2,099 | 476 |
| Original checkpoint, calibrated components 0.65/200 | 0.2073 | 0.2177 | 319 | 851 | 329 |
| Four-epoch run best checkpoint, calibrated components 0.65/200 | **0.2454** | **0.2511** | 322 | 671 | 326 |

The calibrated converter improved paired mean PQ by `+0.1368` (observation-bootstrap 95% CI `+0.1199` to `+0.1533`). Fine-tuning added `+0.0380` (95% CI `+0.0195` to `+0.0575`). The combined change was `+0.1748` (95% CI `+0.1509` to `+0.1985`). These intervals are conditional on one split, one seed, and the selected checkpoint.

The three-epoch follow-up used the calibrated converter during every monitoring pass and a lower learning rate. Its mean monitor PQ values were `0.2930`, `0.2901`, and `0.2864`, all below the starting checkpoint's like-for-like calibrated monitor PQ of `0.3028`. It was therefore rejected; the four-epoch run's `fold1_best.pt` remains selected. Detailed per-stage status, commands, artifacts, and caveats are in the [experiment report](reports/pq_experiment_20260926.md).

Two additional three-epoch runs tested whether reducing the BCE positive-weight cap would suppress false positives. Cap `2.0` peaked at monitor PQ `0.2991`; cap `1.0` peaked at `0.3005`. Both improved over the first follow-up but remained below `0.3028`, so neither was promoted. Their complete epoch curves and prediction overlays are saved in the notebook.

`eval` writes `evaluation_foldF.json` with aggregate and per-image results: PQ/SQ/RQ, pixel and matched-instance overlap metrics, TP/FP/FN, fragmentation and merging counts, and timing. The paired experiment additionally writes calibration and comparison reports under its configured output directory.

Publish measured scores together with the checkpoint, split, support mask, TTA, and post-processing configuration. The draft [technical report](main.tex) still requires completed experiments before its results can be finalized.
