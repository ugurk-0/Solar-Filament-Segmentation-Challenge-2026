# Fold-1 PQ experiment — 26 September 2026

## Outcome

The experiment tested two linked hypotheses: distance-transform watershed was fragmenting elongated filaments, and a corrected mask-training pipeline could improve PQ after instance conversion was fixed. Both effects were measured on fold 1, but this remains a single-split, single-seed local experiment rather than a leaderboard estimate or a controlled ablation of every code change.

The selected checkpoint is `runs/pq_research_20260926/training/fold1_best.pt`, produced at epoch 1 of the four-epoch warm-start run. The selected post-processing is connected components with probability threshold `0.65` and minimum area `200` pixels. The later three-epoch run did not improve the fixed monitor and is retained only as negative evidence.

## Stage status

| Stage | Status | Primary artifact |
|---|---|---|
| Data/support audit | Complete | `runs/pq_research_20260926/data_audit.json` |
| Two-epoch execution smoke test | Complete | `runs/pq_research_20260926/smoke/history.json` |
| Four-epoch warm-start training | Complete | `runs/pq_research_20260926/training/history.json` |
| Twelve-candidate post-processing calibration | Complete | `runs/pq_research_20260926/baseline/calibration_grid.json` |
| Independent baseline assessment | Complete | `runs/pq_research_20260926/baseline/audit.json` |
| Independent trained-checkpoint assessment | Complete | `runs/pq_research_20260926/comparison/audit.json` |
| Three-epoch continuation | Complete; rejected | `runs/pq_research_20260926/training_round2/history.json` |
| Three-epoch positive-cap 2.0 run | Complete; rejected | `runs/pq_research_20260926/training_round3/history.json` |
| Three-epoch positive-cap 1.0 run | Complete; rejected | `runs/pq_research_20260926/training_round4/history.json` |
| Executed presentation notebook | Complete | `notebook.ipynb` |
| Test inference and submission | Pending | — |

The notebook contains saved outputs for the status table, epoch histories, learning curves, prediction/ground-truth overlays, calibration grid, independent assessment, and descriptive full-fold metrics.

## Experimental protocol

Physical observations are grouped by `Path(file_name).stem.split("-")[-1]`, and validation keeps one annotation set per physical observation. Fold 1 contains 141 validation observations divided deterministically into:

- 16 monitoring observations for epoch selection;
- 40 different observations for post-processing calibration;
- 85 untouched observations for the paired assessment.

The approximate projected ±70° longitude support is applied to training loss, predictions, and validation labels. Inference uses no TTA for this experiment and uses a 1024-pixel sliding window. PQ greedily matches instances at IoU ≥ 0.5. The primary estimand is mean per-observation PQ; pooled dataset PQ is reported separately.

The calibration grid contained two instance methods (`watershed`, `components`), three thresholds (`0.35`, `0.50`, `0.65`), and two minimum areas (`50`, `200`). Only the 40 calibration observations selected the converter. The chosen setting was then frozen for both checkpoints on the 85-observation assessment.

## Data audit

On a 48-record training sample containing 396 annotated instances, the corrected support retained 99.986% of annotation pixels versus 81.821% for the previous support. Applying instance conversion to a perfect union mask gave mean PQ 0.9717 for connected components and 0.3683 for watershed. This diagnostic motivated the predeclared converter comparison; it was not itself used as the independent assessment.

## Four-epoch training monitor

The run warm-started the same-fold 60-epoch EMA checkpoint with a fresh optimizer, learning rate `1e-4`, EMA decay `0.9`, and evaluation after every epoch on the fixed 16-observation monitor.

| Epoch | Training loss | Monitor mean PQ |
|---:|---:|---:|
| 1 | 0.2961 | **0.3067** |
| 2 | 0.2882 | 0.2509 |
| 3 | 0.2811 | 0.2471 |
| 4 | 0.2779 | 0.2712 |

The falling loss did not imply improving PQ. Epoch 1 was saved as `fold1_best.pt`. These original monitoring values used the run's then-current converter; the later assessment below rescored checkpoints with one frozen calibrated converter.

## Calibration and independent assessment

The calibration split selected connected components, threshold `0.65`, and minimum area `200`, with calibration mean PQ 0.1676.

| Checkpoint and inference | Mean PQ | Dataset PQ | SQ | RQ | TP | FP | FN | Pixel Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original checkpoint, watershed 0.50/50 | 0.0706 | 0.0706 | 0.5988 | 0.1178 | 172 | 2,099 | 476 | 0.5628 |
| Original checkpoint, components 0.65/200 | 0.2073 | 0.2177 | 0.6202 | 0.3509 | 319 | 851 | 329 | 0.5871 |
| Epoch-1 checkpoint, components 0.65/200 | **0.2454** | **0.2511** | 0.6398 | 0.3924 | 322 | 671 | 326 | 0.6113 |

The post-processing change reduced false positives by 1,248 and increased true positives by 147. On the paired 85 observations, its mean-PQ gain was `+0.1368`; the observation-bootstrap 95% interval was `[+0.1199, +0.1533]`, and the date-block interval was `[+0.1202, +0.1540]`.

With post-processing frozen, the epoch-1 checkpoint added `+0.0380` mean PQ. Its observation-bootstrap 95% interval was `[+0.0195, +0.0575]`, and its date-block interval was `[+0.0189, +0.0571]`. Relative to the original checkpoint and watershed default, the combined paired gain was `+0.1748`, with observation-bootstrap interval `[+0.1509, +0.1985]`.

These intervals measure paired variation within this one assessment split. They do not include uncertainty from alternative training seeds, folds, temporal partition choices, model selection, or the Kaggle test distribution.

## PQ-guided follow-up

The second run started from the selected epoch-1 checkpoint and used three epochs, learning rate `2e-5`, EMA decay `0.99`, and the frozen calibrated converter for every monitoring pass. The like-for-like calibrated monitor PQ of its starting checkpoint was 0.3028.

| Epoch | Training loss | Monitor mean PQ | Decision |
|---:|---:|---:|---|
| 1 | 0.2763 | 0.2930 | Below start |
| 2 | 0.2771 | 0.2901 | Below start |
| 3 | 0.2766 | 0.2864 | Below start |

The follow-up did not beat its starting checkpoint and was rejected. This result argues against continuing the same fine-tuning trajectory merely because loss remains low. Future training changes should be evaluated with the frozen calibrated converter and should target the remaining recognition errors, especially false positives and missed instances.

### False-positive-focused follow-ups

Two more isolated runs started from the same selected checkpoint with learning rate `1e-5`, EMA decay `0.99`, and the same calibrated converter and 16-observation monitor. Only the BCE positive-weight cap changed.

| Run | Positive-weight cap | Epoch PQ sequence | Best epoch | Best PQ | Mean TP | Mean FP | Mean FN | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Round 3 | 2.0 | 0.2945, 0.2991, 0.2938 | 2 | 0.2991 | 4.375 | 7.188 | 3.062 | Reject |
| Round 4 | 1.0 | 0.2971, 0.3005, 0.2870 | 2 | 0.3005 | 4.312 | 7.062 | 3.125 | Reject |

Reducing positive weighting moved the best monitor PQ toward the 0.3028 starting value and modestly reduced false positives, but neither run exceeded the starting checkpoint. Round 4's lower loss (`0.2566` at its best epoch) did not translate into higher PQ. The result supports exploring false-positive control through a more targeted objective or sampling change, but it does not support replacing the selected checkpoint.

## Commands run

The verified interpreter was `C:/Users/ugurk/AppData/Local/Programs/Python/Python312/python.exe` with PyTorch 2.4.1+cu124 and CUDA available.

```powershell
python -m pytest tests/test_solution_regressions.py -q
python solution.py train --fold 0 --epochs 2 --limit 8 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/pq_research_20260926/smoke --no-tta --window 512 --crop-size 384 --eval-every 1 --preview-count 1
python audit_data.py
python pq_experiment.py baseline --checkpoint runs/corrected_loss_fold1/fold1.pt --output runs/pq_research_20260926/baseline
python solution.py train --fold 1 --epochs 4 --data-dir MAGFiLO_1.0_Kaggle_2026/train --work-dir runs/pq_research_20260926/training --init-ckpt runs/corrected_loss_fold1/fold1.pt --eval-every 1 --eval-max-images 16 --preview-count 3 --no-tta --window 1024 --lr 0.0001 --ema-decay 0.9
python pq_experiment.py compare --checkpoint runs/pq_research_20260926/training/fold1_best.pt --output runs/pq_research_20260926/comparison --baseline runs/pq_research_20260926/baseline/audit.json --parameters runs/pq_research_20260926/baseline/selected_postproc.json
python run_training_round2.py
python run_training_round3.py
python run_training_round4.py
python build_notebook.py
```

The final notebook was executed in place with `nbclient` and validated with `nbformat`. The final regression run passed 12 tests. Existing checkpoints were never overwritten; every training cycle used its own work directory.

## Remaining work

Test inference and submission generation are still pending. A stronger estimate also requires more folds or a temporal-block validation design, additional seeds, and controlled ablations of the support, loss, crop-sampling, and optimization changes. Test labels must not be used for post-processing or checkpoint selection.
