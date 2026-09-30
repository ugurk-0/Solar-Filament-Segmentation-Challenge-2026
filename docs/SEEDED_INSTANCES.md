# PQ and seeded mask growth

Panoptic Quality (PQ) evaluates individual filament masks. Predicted and annotated
instances are greedily matched by decreasing intersection-over-union (IoU), with
IoU at least 0.5 and each instance used at most once.

`PQ = SQ × RQ`, where `SQ` is the average IoU of matched masks and
`RQ = TP / (TP + FP/2 + FN/2)`. A perfect prediction scores 1. PQ is not pixel
accuracy: a merged pair, a fragmented filament, an extra detection, or a missing
filament can hurt instance matching even when many foreground pixels are correct.

The retained baseline has mean per-observation PQ **0.2714** on 85 local research
observations. Pooled PQ is **0.2710**, comprising SQ **0.6428** and RQ **0.4216**.
Mean PQ averages each observation's score equally; pooled PQ aggregates matches
and errors across observations before computing the ratio. The last user-reported
Kaggle score is **0.24**, a different measurement. The local assessment has been
reused across experiments and is not an untouched test set.

## Hypothesis and implementation

The previous instance-quality classifier removed too many correct detections.
This experiment instead changes mask geometry while keeping the neural network
frozen. It starts with connected regions above probability 0.8 containing at least
500 pixels, then grows them through connected pixels above a lower threshold.
Weak regions with no strong seed are discarded.

- Merge mode retains a whole low-threshold component if it contains a seed.
- Separate mode uses a marker-based watershed of negative probability to grow
  seeds while maintaining separate ownership where they meet.

The fixed search compares lower thresholds 0.50, 0.65, and 0.75 in both modes,
plus the unchanged baseline. Masks stay inside annotation support. Cleanup checks
ownership before filling holes, and final instances must have at least 500 pixels.
These choices can recover edges, but can also merge true objects or split a single
filament with disconnected confident cores. Measurement determines whether to
retain the change.

## Validation and reproduction

```bash
python -m pytest tests/test_seeded_instances.py -q
python experiment_seeded_instances.py --smoke
python experiment_seeded_instances.py
```

The smoke run uses two calibration and two assessment observations and does not
establish model quality. Full calibration selects a setting on 40 observations,
freezes it, and compares it with the baseline on the same 85 research observations.
Pooled baseline PQ must reproduce within 1e-9. Promotion requires improved pooled
PQ and a positive lower bound for the paired mean-PQ difference's exploratory
95% bootstrap interval. No test labels are used.

The runner requires the existing Round-5 checkpoint and validation probability
cache. It verifies checkpoint identity and inference configuration; protocol files
record source hashes, annotation hash, candidates, and observation IDs. Artifacts
are isolated under `runs/seeded_instances_20260930/`; completed assessments are
preserved. The experiment does not automatically replace production defaults or
export a submission. There is no training change, so retraining the unchanged
U-Net is unnecessary for this comparison.

Implementation: [seeded_instances.py](../seeded_instances.py) and
[experiment_seeded_instances.py](../experiment_seeded_instances.py).
