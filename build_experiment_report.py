"""Write documentation from completed, measured experiment artifacts only."""
import json
from pathlib import Path
import shutil


ROOT = Path("runs/pq_research_20260926")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    data = read("data_audit.json")
    old = read("baseline/audit.json")
    new = read("comparison/audit.json")
    full_old = read("baseline/full_fold.json")["summary"]
    full_new = read("comparison/full_fold.json")["summary"]
    history = read("training/history.json")
    submission = read("submission/submission_manifest.json")
    grid = read("baseline/calibration_grid.json")
    selected = read("baseline/selected_postproc.json")
    best = max(history, key=lambda r: r["pq"] if r["pq"] is not None else -1)
    rows = [("Old EMA + watershed", old["baseline"]),
            ("Old EMA + calibrated post-processing", old["selected"]),
            ("Fine-tuned EMA + same calibrated post-processing", new["selected"])]
    table = "| Checkpoint / processing | n | Mean PQ | Pooled PQ | SQ | RQ | TP | FP | FN | Pixel Dice |\n"
    table += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for label, row in rows:
        table += f"| {label} | {row['n']} | {row['mean_pq']:.6f} | {row['dataset_pq']:.6f} | {row['sq']:.6f} | {row['rq']:.6f} | {row['tp']} | {row['fp']} | {row['fn']} | {row['pixel_dice']:.6f} |\n"
    epochs = "| Epoch | Loss | Monitor PQ | Seconds |\n|---|---:|---:|---:|\n"
    for row in history:
        epochs += f"| {row['epoch']} | {row['loss']:.6f} | {row['pq']:.6f} | {row['elapsed_seconds']:.1f} |\n"
    text = f"""# PQ experiment — 2026-09-26

## Question and measured conclusion

Does replacing distance-transform watershed with connected components reduce fragmentation,
and does fine-tuning after correcting crop sampling, losses and support improve PQ further?
The primary endpoint is mean image PQ, with one annotation record per physical observation.
All comparisons below use the same fold, support, inference settings, matching rule and
checkpoint semantics (EMA). This is a single-seed exploratory experiment.

{table}

The historical full-fold mean PQ **0.075559** (pooled PQ **0.072281**) came from the old
wedge support, unmasked ground truth and eight-view TTA. It is **not directly comparable**
to the new protocol. The table re-evaluates the old checkpoint fairly under the new protocol.

## Why the old pipeline failed

- The polar-angle limb mask removed equatorial pixels. On 48 randomly selected training
  annotation records ({data['n_instances']} instances), it retained
  **{data['old_retained_annotation_pixels']:.3%}** of annotated pixels, versus
  **{data['new_retained_annotation_pixels']:.3%}** for the longitude projection.
- With perfect foreground-union masks on that training-only sample, watershed achieved mean
  PQ **{data['perfect_union_postprocess_mean_pq']['watershed']:.6f}**, versus
  **{data['perfect_union_postprocess_mean_pq']['components']:.6f}** with connected components.
  This is an oracle diagnostic of post-processing, not a model score or universal upper bound.
- Positive crop x/y coordinates previously came from independently selected filament pixels.
- clDice previously eroded structures away rather than accumulating a morphological skeleton,
  and used mismatched precision/sensitivity denominators. The replacement has identity,
  broken-line, masking and finite-gradient checks. It remains a surrogate, not direct PQ loss.
- EMA evaluation previously overwrote the live model, invalidating continued Adam training.
  The old 60-epoch runs skipped validation, so this bug did not explain their particular score;
  it would have affected the requested training with validation enabled.
- Checkpoints were overwritten every epoch without preserving the best measured PQ. Skipped
  validation printed zero or a stale best score. New logs use `not_evaluated` when appropriate.

## Mathematical protocol

The approximate support is the disk intersected with
`abs(x-cx) <= sin(70 degrees) * sqrt(R^2 - (y-cy)^2)`, with R=950 and solar tilt B0=0.
It follows from orthographic projection x=R cos(latitude) sin(longitude),
y=R sin(latitude). No per-image WCS or tilt estimate was fitted.
Masks, losses and inference use the same support. Ground-truth masks entirely outside support
are excluded. Every retained instance keeps its identity, even if support clipping disconnects it.

Greedy matching accepts IoU >= 0.5. For each image, PQ = sum(matched IoUs) /
(TP + FP/2 + FN/2). Empty/empty PQ is zero; empty/empty pixel Dice is one.
Mean image PQ is the primary estimand. Pooled PQ aggregates counts and IoUs first and therefore
weights observations differently. SQ and RQ in the comparison table are the pooled components.

## Experimental design and uncertainty

Original fold 1 is preserved using GroupKFold by physical observation. There are 16 fixed
monitor images for epoch selection, 40 different calibration images, and 85 assessment images.
`protocol.json` and `split.json` save exact IDs; checkpoint SHA256 identifies evaluated weights.
The original fold was already inspected by prior agents, so the assessment is locally held out
from this round's parameter selection, not pristine confirmatory evidence.

The calibration grid has {len(grid)} candidates: 2 methods × 3 thresholds × 2 minimum areas.
Selection on calibration mean PQ yields `{selected['parameters']}`. Parameters are frozen for
the paired old/new comparison. The fine-tuning checkpoint is selected on monitor PQ, epoch
**{best['epoch']}**; the assessment images never select the epoch.

Post-processing effect, paired observation bootstrap:
`{old['paired']}`

Training effect with identical calibrated processing:
`{new['paired_training_effect']}`

Total effect relative to old weights and watershed:
`{new['paired_total_effect']}`

Intervals are percentile intervals (10,000 observation resamples, 3,000 date-block resamples,
seed 2026). Date blocks address within-day clustering only. Exact-image grouping does not
eliminate correlated structures on adjacent days. Intervals condition on this training seed,
split and selection procedure; they are not confidence intervals for leaderboard performance.
Individual code fixes were bundled; no causal attribution to one training fix is established.

## Training and full-fold results

The existing fold-1 60-epoch EMA is warm-started for four full-data epochs with a fresh AdamW
optimizer, lr=1e-4, cosine decay, EMA decay=0.9, 768-pixel crops, effective batch size 8,
and BCE/Dice/clDice weights 0.3/0.3/0.4. Positive BCE weighting is capped at 5 (formerly 20).
No external weights or non-mask auxiliary labels were introduced. Inference uses 1024-pixel
windows, overlap 256 and no TTA in every new comparison. Full-fold metrics include images
used for selection and are explicitly descriptive:

- Old weights, calibrated processing: `{full_old}`
- Fine-tuned weights, calibrated processing: `{full_new}`

{epochs}

Visual artifacts: [live dashboard](../runs/pq_research_20260926/training/dashboard.html),
[learning curves](../runs/pq_research_20260926/training/learning_curves.png),
[fixed prediction examples](../runs/pq_research_20260926/comparison/previews).
The notebook embeds actual saved metrics and instance overlays.

## Files and commands

Changed: `solution.py`, `tests/test_solution_regressions.py`, `requirements.txt`,
`notebook.ipynb`, `README.md`, `main.tex`. Added: `pq_experiment.py`, `audit_data.py`,
`build_notebook.py`, `build_experiment_report.py`, and reports. Historical source/notebook/report
backups and all generated artifacts are under `runs/pq_research_20260926`; old checkpoints
remain untouched.

Python used: `C:/Users/ugurk/AppData/Local/Programs/Python/Python312/python.exe`.
The shell did not have a `python` executable on PATH. In the commands below, substitute that
absolute executable if necessary. Pinned dependencies were installed; scikit-learn was
corrected from 1.6.0 to the pinned 1.5.2. Torch is 2.4.1+cu124 on an RTX 3060 laptop GPU.
The shared environment's `pip check` reports pre-existing conflicts in unrelated OpenCV/siuba;
neither is imported by this pipeline. Use a dedicated environment for a clean reproduction.

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/test_solution_regressions.py -q
python solution.py train --fold 0 --epochs 2 --limit 8 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/pq_research_20260926/smoke --no-tta --window 512 --crop-size 384 --eval-every 1 --preview-count 1
python audit_data.py
python pq_experiment.py baseline --checkpoint runs/corrected_loss_fold1/fold1.pt --output runs/pq_research_20260926/baseline
python solution.py train --fold 1 --epochs 4 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/pq_research_20260926/training --init-ckpt runs/corrected_loss_fold1/fold1.pt --eval-every 1 --eval-max-images 16 --preview-count 3 --no-tta --window 1024 --lr 0.0001 --ema-decay 0.9
python pq_experiment.py compare --checkpoint runs/pq_research_20260926/training/fold1_best.pt --output runs/pq_research_20260926/comparison --baseline runs/pq_research_20260926/baseline/audit.json --parameters runs/pq_research_20260926/baseline/selected_postproc.json
python build_notebook.py
python build_experiment_report.py
```

Training rejects an existing checkpoint in its work directory. Change work directories for
new experiments rather than overwriting valuable runs. The notebook includes the exact
submission call, carrying the frozen post-processing parameters.

## Submission and remaining scientific limits

The generated CSV covers **{submission['n_images']} test images**, with
**{submission['n_instances']} predicted instances**. All RLE round trips passed. Per-image
instance counts, including zero-prediction images, are saved in `submission_manifest.json`.
No Kaggle upload or leaderboard score is claimed. No post-processing was tuned on test labels.
The submission uses the monitor-selected fine-tuned checkpoint and calibrated parameters.

This completes one reproducible improvement experiment and a local submission artifact,
not a proof of optimal challenge performance. Remaining work for a stronger competitive
estimate includes independent training seeds, additional folds, temporal-block validation,
and controlled architecture/pretraining ablations with a fixed selection budget.

## Sources and skills

- [MAGFiLO annotation protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437148/).
- [clDice authors' implementation](https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch).
- [Skills review](skills_review.md): repository popularity, relevance and evidence limitations.
- Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065),
  acknowledged for scientific-critical-thinking guidance applied to the experimental design.
"""
    Path("reports/pq_experiment_20260926.md").write_text(text, encoding="utf-8")
    for name in ("README.md", "main.tex"):
        backup = ROOT / ("before_" + name)
        if not backup.exists(): shutil.copyfile(name, backup)
    readme = f"""# Solar Filament Segmentation Challenge 2026

Reproducible, mask-only segmentation optimized and selected for Panoptic Quality.

The latest experiment corrected annotation support, crop sampling, clDice, EMA validation,
and instance fragmentation. On the same {new['selected']['n']}-observation assessment subset,
mean PQ changed from **{old['baseline']['mean_pq']:.4f}** (old weights + watershed) to
**{new['selected']['mean_pq']:.4f}** (fine-tuned EMA + calibrated processing).
This is a local single-fold experiment; no leaderboard score is claimed.

- [Measured results, uncertainty, exact commands and limitations](reports/pq_experiment_20260926.md)
- [Notebook with metrics and prediction examples](notebook.ipynb)
- [Training dashboard](runs/pq_research_20260926/training/dashboard.html)
- [Scientific skills review](reports/skills_review.md)
- [Technical report](main.tex)
- [Generated submission](runs/pq_research_20260926/submission/submission.csv)

## Reproduction

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/test_solution_regressions.py -q
```

Use a CUDA-enabled Torch 2.4.1 build for training. The verified environment uses
`2.4.1+cu124`. If `python` is not on PATH on this machine, use
`C:/Users/ugurk/AppData/Local/Programs/Python/Python312/python.exe`.

Open `notebook.ipynb` from the repository root. By default it displays completed artifacts.
Its explicit stage switches run training, calibration, assessment and submission separately.
Use a **new work directory** to retrain; existing checkpoints are protected against overwrite.
Submission generation does not train and does not upload to Kaggle.

`solution.py` owns data loading, training, inference, metrics and post-processing.
`pq_experiment.py` owns the fixed monitoring/calibration/assessment split and paired bootstrap.
The notebook is a presentation layer over these modules. Exact reproduction commands and
the warm-start checkpoint provenance are in the experiment report.

## Evaluation contracts

- Group by the physical observation suffix, never COCO annotation ID; deduplicate validation.
- Apply the same approximate longitude support to loss, inference and ground truth.
- Greedy instance matching at IoU >= 0.5; primary metric is mean image PQ.
- Monitor selects epochs, calibration selects post-processing, assessment measures paired effects.
- Full-fold results include selection images and are descriptive. Preserve seed, support,
  TTA, window geometry and checkpoint semantics when comparing experiments.
- Test labels are not used for tuning. RLE serialization is checked by decoding every mask.

The original 60-epoch checkpoints and prior artifacts are preserved. Historical reports
under `runs/` describe superseded protocols; use the dated experiment report for current results.
Datasets, checkpoints, probability caches and generated outputs are excluded by `.gitignore`.
"""
    Path("README.md").write_text(readme, encoding="utf-8")
    latex_rows = "\n".join(f"{label.replace(' + ', ' / ')} & {r['mean_pq']:.4f} & {r['dataset_pq']:.4f} & {r['tp']} & {r['fp']} & {r['fn']} \\\\" for label,r in rows)
    latex = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,booktabs,graphicx,hyperref}
\title{Diagnosing and Improving Panoptic Quality in Solar Filament Segmentation}
\author{Solar Filament Segmentation Challenge 2026}
\date{26 September 2026}
\begin{document}\maketitle
\begin{abstract}
We audit a mask-only U-Net pipeline on MAGFiLO. A support-geometry error removed
annotated equatorial pixels, and distance-transform watershed fragmented elongated
instances. We compare old weights, calibrated instance conversion, and a four-epoch
fine-tuning run under one common local protocol. Results are exploratory and single-seed;
no leaderboard performance or optimality is claimed.
\end{abstract}
\section{Data and evaluation}
Images are grouped by physical observation before five-fold GroupKFold splitting;
validation retains one annotation set per observation. Within fold 1, fixed disjoint
subsets contain 16 monitoring, 40 calibration, and 85 assessment observations.
Only the first two subsets select checkpoints and post-processing, respectively.
Earlier agents inspected aggregate metrics on this fold, so this is not a pristine
confirmatory holdout. Temporal correlation may persist across distinct observations.

Predictions and labels are restricted to the same approximate support,
$|x-c_x|\leq\sin(70^\circ)\sqrt{R^2-(y-c_y)^2}$ inside a radius-$R$ disk,
with $R=950$ and zero solar tilt. This is the orthographic longitude projection,
not the former polar-angle wedge. Annotation support is approximate, not a WCS solution.
Greedy matches have IoU $\geq0.5$ and
$\mathrm{PQ}=\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)/(TP+FP/2+FN/2)$.
We report both mean per-image PQ and pooled-count PQ. Empty/empty image PQ is zero.

\section{Method and diagnostics}
A ResNet-18 encoder and attention-gated U-Net decoder use GroupNorm and a residual
dilated bottleneck. Training uses segmentation polygons only, with no spine or chirality
targets. Positive crop coordinates now come from one sampled foreground point.
The loss combines weighted BCE, soft Dice and morphological soft clDice with weights
0.3, 0.3 and 0.4. BCE positive weighting is capped at 5; clDice is a topology surrogate,
not direct optimization of discrete PQ. EMA validation operates on a copy and cannot
overwrite live optimizer parameters. The best monitor-PQ checkpoint is saved separately.
"""
    latex += f"""
On 48 training-only annotation records ({data['n_instances']} instances), retained annotation
pixels increased from {100*data['old_retained_annotation_pixels']:.3f}\\% to
{100*data['new_retained_annotation_pixels']:.3f}\\%. Converting perfect foreground unions
to instances yielded mean PQ {data['perfect_union_postprocess_mean_pq']['watershed']:.4f}
with watershed and {data['perfect_union_postprocess_mean_pq']['components']:.4f} with
connected components. These are post-processing diagnostics, not model scores.
"""
    latex += r"""
\section{Experiment}
We warm-start the existing same-fold 60-epoch EMA for four full-data epochs with a
fresh AdamW optimizer, learning rate $10^{-4}$, cosine scheduling and EMA decay 0.9.
Crops are $768^2$ pixels, with effective batch size 8. Inference uses 1024-pixel windows,
256-pixel overlap and no TTA in all paired comparisons. Calibration compares two
instance methods, three thresholds and two minimum areas. The selected parameters
are frozen for the old/new weight comparison. Individual training repairs are bundled;
this experiment cannot identify their separate causal contributions.
\begin{center}\small
\begin{tabular}{lrrrrr}\toprule
Variant & Mean PQ & Pooled PQ & TP & FP & FN\\\midrule
""" + latex_rows + r"""
\\\bottomrule\end{tabular}\end{center}
"""
    effect = new["paired_training_effect"]
    latex += f"""
The paired mean-PQ training effect is {effect['mean_pq_delta']:.4f}, with a percentile
95\\% observation-bootstrap interval
[{effect['observation_bootstrap_95ci'][0]:.4f}, {effect['observation_bootstrap_95ci'][1]:.4f}]
and date-block sensitivity interval
[{effect['date_block_bootstrap_95ci'][0]:.4f}, {effect['date_block_bootstrap_95ci'][1]:.4f}].
There are 10,000 observation resamples and 3,000 date-block resamples, seed 2026.
These intervals condition on the split and training seed and do not quantify leaderboard uncertainty.
The chosen checkpoint is epoch {best['epoch']} by monitoring PQ.
Full-fold descriptive mean PQ is {full_new['mean_pq']:.4f}; it includes selection images.
"""
    latex += r"""
\section{Reproducibility and limits}
The repository contains pinned dependencies, focused regression tests, split IDs,
checkpoint hashes, per-image metrics, training curves, instance overlays and an executed
presentation notebook. A local CSV is generated with every RLE verified by decoding.
No test labels are used for tuning and no Kaggle score is claimed. Further validation
requires independent seeds, more folds and temporal blocks. The skill-based scientific
review procedure is acknowledged in the experiment report\cite{kassis2026}; its authors
report no task-level evaluation, so no agent-performance gain is inferred from popularity.
\begin{thebibliography}{9}
\bibitem{magfilo} Ahmadzadeh et al. A dataset of manually annotated filaments from
H-alpha observations. Scientific Data (2024). doi:10.1038/s41597-024-03876-y.
\bibitem{cldice} Shit et al. clDice: a novel topology-preserving loss function for
tubular structure segmentation. CVPR (2021). Implementation: github.com/jocpae/clDice.
\bibitem{pq} Kirillov et al. Panoptic Segmentation. CVPR (2019).
\bibitem{kassis2026} Kassis, Agarwal, He, Patel, Brueckner. Scientific Agent Skills:
A Library of Procedural Knowledge for Research Agents (2026).
doi:10.48550/arXiv.2609.00065.
\end{thebibliography}\end{document}
"""
    Path("main.tex").write_text(latex, encoding="utf-8")
    print("Wrote measured experiment report, README.md and main.tex")


if __name__ == "__main__":
    main()
