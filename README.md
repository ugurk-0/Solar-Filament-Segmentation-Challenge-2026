# Solar Filament Segmentation Challenge 2026 — Solution

Segmentation of solar filaments in GONG H-alpha observations (MAGFiLO v1.0),
optimised for the **Panoptic Quality** metric (PQ = SQ × RQ, IoU match @ 0.5).

Competition: https://www.kaggle.com/competitions/filament-segmentation-2026

## Project status

This is a work-in-progress research solution. `main.tex` is a draft report;
ablation results and a final competition score have not been established.
Datasets, checkpoints, and generated results are excluded from Git. Download
the competition data separately and configure local paths before running.

The notebook presents the paired PQ experiment in `pq_experiment.py`.
It displays locally saved artifacts when available and leaves expensive stages
disabled by default. Use `RUN_TRAINING`, `RUN_CALIBRATION`, `RUN_ASSESSMENT`,
and `RUN_SUBMISSION` to enable those stages. The configured baseline checkpoint
must exist before training or calibration; it is not distributed here.
`audit_data.py` audits annotation support, and `build_notebook.py` generates
the presentation notebook.

Run the regression suite after installing dependencies:

```bash
python -m pytest tests/test_solution_regressions.py -q
```

## Repository layout

```
filament-challenge-2026/
├── README.md            # this file
├── requirements.txt     # pinned dependencies
├── solution.py          # full pipeline (train / eval / tune / submit)
├── notebook.ipynb       # notebook illustrating the entire pipeline (competition requirement)
├── main.tex             # 4-page technical report (Overleaf template)
├── tests/               # focused metric and regression tests
├── .github/agents/      # project-specific Copilot agent guidance
└── runs/                # local checkpoints and evaluation artifacts (not committed)
```

## Quick start

```bash
pip install -r requirements.txt

# For NVIDIA GPU runs, use CUDA wheels matching the installed driver.
# The verified local environment uses the cu124 wheels.
python -m pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# 0. smoke test (2 epochs, 8 images)
python solution.py train --fold 0 --epochs 2 --limit 8

# 1. final training for one fold
python solution.py train --fold 0 --epochs 60 --work-dir runs/fold0

# 2. local Panoptic Quality per fold
python solution.py eval --fold 0 --work-dir runs/fold0
# writes runs/fold0/evaluation_fold0.json with PQ, Dice, IoU, relations, and timing

# 3. tune post-processing on the held-out fold
python solution.py tune --fold 0 --work-dir runs/fold0

# 4. ensemble folds -> submission.csv
python solution.py submit --ckpts runs/fold0/fold0.pt runs/fold1/fold1.pt ...
```

The `train` command only trains. Run `eval`, `tune`, and `submit` separately,
or execute the notebook for the complete workflow. On Kaggle, paths point to `/kaggle/input/...` by default; override with
`--data-dir` / `--test-dir` / `--work-dir` for local runs.

## Method (see main.tex for details)

| Stage | Choice | Rationale |
|---|---|---|
| Split | GroupKFold on **physical observation name** | same image annotated by 2–3 annotators; COCO-id split leaks |
| Limb mask | ±70° of central meridian | annotations stop there; FP otherwise |
| Model | timm-encoder U-Net, attention gates, dilated bottleneck | thin structures need full-res skips + wide context |
| Training targets | segmentation masks only | complies with the competition restriction on other ground-truth metadata |
| Loss | BCE + soft-Dice + soft-clDice | clDice penalises fragmented filament predictions |
| Train | filament-centred crops, dihedral aug, AMP, cosine, EMA | ~2% positives; disk is isotropic |
| Inference | sliding window + 8-way dihedral TTA (+ fold ensemble) | overlap-add probability averaging |
| Instances | connected components by default; optional EDT watershed | compare fragmentation and merging on held-out observations |
| Tuning | grid on **local PQ** | the only metric that mirrors the leaderboard |

## Reproducibility

- `python solution.py train --fold 0 --epochs 2 --limit 8` smoke-tests the training path on a handful of images.
- Run `eval`, `tune`, and `submit` as separate commands after training; they are never triggered implicitly by `train`.
- Checkpoints, tuned post-processing (`best_postproc.json`) and `submission.csv` are written under `CFG.work_dir`.
- Notebook `notebook.ipynb` re-executes every stage against `solution.py`.

## Final submission workflow

Training and submission are separate operations. Train final folds on a GPU, preserve the resulting checkpoints, then run inference with those checkpoints. The submission run must not retrain the model.

```bash
# Run once per fold on a GPU; use a separate work directory for each fold.
python solution.py train --fold 0 --epochs 60 --work-dir runs/fold0
python solution.py train --fold 1 --epochs 60 --work-dir runs/fold1
python solution.py train --fold 2 --epochs 60 --work-dir runs/fold2
python solution.py train --fold 3 --epochs 60 --work-dir runs/fold3
python solution.py train --fold 4 --epochs 60 --work-dir runs/fold4

# Tune post-processing on a held-out training fold, never on test images.
python solution.py tune --fold 0 --work-dir runs/fold0

# Generate the single CSV required by the competition.
python solution.py submit --work-dir runs/final --ckpts runs/fold0/fold0.pt runs/fold1/fold1.pt runs/fold2/fold2.pt runs/fold3/fold3.pt runs/fold4/fold4.pt
```

The notebook skips final training unless `RUN_TRAINING` is explicitly set. It rejects smoke-test checkpoints (`--limit 8`, two epochs) for submission. The generated CSV must contain exactly `filament_id` and `segmentation_rle`, with unique filament IDs and one row per predicted instance. The final evaluation package also requires this public repository, pinned `requirements.txt`, the complete reproducible notebook, and the technical report through the organizer's final-submission process. Training uses segmentation masks only; do not re-enable the optional spine target unless the organizers explicitly confirm that auxiliary ground-truth metadata is allowed.

## Scores (local validation)

| Fold | Epochs | Mean PQ | Dataset PQ | Dataset SQ | Dataset RQ | Pixel Dice | Pixel IoU | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | Pending corrected TTA evaluation | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

The original fold-1 evaluation exposed a transpose-plus-flip TTA inversion
defect, so its metrics were superseded. The corrected 8-way TTA evaluation is
in progress. Do not use this row for model selection until its JSON report is
written and reviewed.

## Ablation Plan

| stage | local PQ |
|---|---|
| baseline U-Net (BCE+Dice) | Not measured |
| + clDice mask loss | Not measured |
| + attention gates + dilated bottleneck | Not measured |
| + TTA + fold ensemble + tuned post-proc | Not measured |

These values are intentionally not estimated. The corrected fold-1 full-fold
evaluation must finish before any score is reported. Do not report a
leaderboard score as a local PQ measurement.

## Evaluation outputs

`python solution.py eval --fold F --work-dir PATH` writes
`PATH/evaluation_foldF.json`. Its `aggregate` section measures:

- `pq`, `sq`, and `rq`: Panoptic Quality and its standard components;
- `pixel_dice` and `pixel_iou`: foreground-union Dice and IoU distributions;
- `mean_instances_dice` and `mean_instances_iou`: matched-instance score distributions;
- `one_to_many_gt`: ground-truth instances touched by at least two predictions;
- `many_to_one_pred`: predictions touching at least two ground-truth instances;
- `tp`, `fp`, and `fn`: instance detection counts;
- `elapsed_seconds` and `images_per_second`: end-to-end validation efficiency.

The `per_image` section preserves the same values for every validation image,
including matched-instance score lists. This makes the required score
distributions and fragmentation/merging analysis reproducible rather than just
reporting a single mean.

For CPU-only smoke evaluation, use one annotated native-resolution image without TTA:

```bash
python solution.py eval --fold 0 --positive-only --limit 1 --no-tta --window 2048
```

The current CPU smoke report took 200.86 seconds (0.00498 images/second) on a
validation image with zero ground-truth instances. Its zero overlap scores are
therefore an empty-image smoke result, not a model-quality estimate. Use the
full validation fold on a GPU for the meaningful score distributions.

## Verified smoke-test results

The current notebook successfully verified the following local data facts:

| Check | Result |
|---|---:|
| COCO annotation entries | 1,154 |
| Unique physical observations | 707 |
| Observations with duplicate annotator records | 296 |
| Annotated limb-region coverage | 26.3% |
| Smoke-training configuration | fold 0, 2 epochs, 8 images |
| Completed final model checkpoints | folds 0 and 1, 60 epochs each |
| Submission CSV | Not generated yet |
