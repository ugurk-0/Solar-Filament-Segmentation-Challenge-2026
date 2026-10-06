# YOLO audit and continuation — 6 October 2026

No Python training process was active when work resumed. Both October 3 jobs
had completed. The README had not yet incorporated their measured results.

| Actual experiment | Completed / selected epoch | Mean assessment PQ | Pooled assessment PQ |
|---|---:|---:|---:|
| YOLO11n, batch 2, 20-epoch budget | 20 / 17 | 0.319854 | 0.331939 |
| YOLO11n, batch 1, 50-epoch budget, mislabeled as medium | 17 / 11 | 0.316368 | 0.332027 |
| Previous U-Net + skeleton merging | No new training | 0.289128 | 0.287762 |

The retained nano candidate used confidence 0.5 and minimum area 128, selected
on 40 calibration observations. Assessment used the same 85 reused observations,
without TTA. Its gain over skeleton merging is 0.030725 mean PQ, with paired
observation-bootstrap 95% interval [0.005056, 0.056648]. TP/FP/FN are 291/195/357.
This is local research evidence, not a new Kaggle score. The user-reported
leaderboard score remains 0.24; PQ 0.60 has not been achieved.

## Model identity correction

`experiment_yolo11m.py` declared a medium model but loaded
`pretrained/yolo11n-seg.pt`. Both saved checkpoints report scale `n`, architecture
`yolo11n-seg.yaml`, and 2,842,803 unfused parameters. Their initialization hashes
also match. The old report now carries a correction while preserving its
original recorded numbers. No claim about medium-model performance is supported.

The weight path is fixed. New runs use `train_yolo_experiment.py`, which verifies
the loaded task and architecture scale before training and again for each
monitored checkpoint. A regression test rejects nano weights when medium is
requested. The old experiment directory remains preserved.

Understand-Anything change-review guidance was consulted. No saved knowledge
graph exists here; conclusions were verified against source, checkpoint
metadata, and run artifacts. `/understand` can build the optional graph.

## Cheap refinement trial

The pending eight-setting YOLO/U-Net probability refinement grid was completed
on the 40 calibration observations. Best setting: radius 8, U-Net probability
0.8, YOLO confidence 0.3, minimum area 128.

| Calibration result | Mean PQ | Pooled PQ | TP / FP / FN |
|---|---:|---:|---:|
| Direct calibrated nano | 0.285366 | 0.288944 | 111 / 112 / 158 |
| Best probability refinement | 0.286476 | 0.280376 | 113 / 127 / 156 |

The mean gain is too small for the predeclared 0.005 gate, and pooled PQ falls.
Decision: **rejected on calibration**, without a new assessment evaluation.
See [full grid](yolo_refinement_20261006.md). This tests probability-map
refinement, not a separately trained crop refiner.

## Next training hypothesis and protocol

Fine-tune the retained nano epoch-17 checkpoint with a fresh optimizer and a
lower starting learning rate (0.00015). Keep resolution 1536, batch 2, nominal
batch 8, the same support-aware loss, seed, observation split, and no TTA.
Budget: 12 epochs; at least 6, then stop after 5 epochs without a monitoring
gain greater than 0.001. Preserve the original parent as a monitoring control.
Only a monitoring winner advances to calibration and assessment against the
retained nano, not the weaker historical U-Net baseline.

Preparation reuses the validated observation dataset. Monitoring reconstructs
only masks with confidence at least 0.1; calibration uses 0.05, the lowest
tested threshold. Lower-scored masks were always discarded by scoring, so
their expensive native-resolution reconstruction is unnecessary. Initial
control inference uses a separate model object: Ultralytics prediction can
fuse convolution/batch-normalization layers in place, and those fused weights
must not replace the unfused training initialization.

```powershell
python -m pytest tests/test_yolo_identity.py tests/test_yolo_pipeline.py tests/test_yolo_refinement.py -q
python train_yolo_experiment.py --variant n --weights runs/yolo11n_20261003/best_pq.pt --output runs/yolo11n_finetune_20261006 --epochs 12 --lr0 0.00015 --batch 2 --smoke
python train_yolo_experiment.py --variant n --weights runs/yolo11n_20261003/best_pq.pt --output runs/yolo11n_finetune_20261006 --epochs 12 --lr0 0.00015 --batch 2
```

Use the separate interpreter
`C:/Users/ugurk/AppData/Local/FilamentYolo/venv/Scripts/python.exe` on this machine.
The focused checks passed 11 tests; the full suite passed **65 tests**. The
fine-tuning smoke test completed training, calibration, and local evaluation.
The full-resolution parent control reproduced monitoring PQ **0.3417497475**
exactly with the optimized inference threshold. The new run's state/report
records actual progress and results; its outcome must not be inferred from
the hypothesis.
