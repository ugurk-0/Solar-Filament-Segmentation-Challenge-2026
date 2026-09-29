# Instance-quality model

The existing U-Net predicts foreground probabilities. High probability alone
does not establish that a connected region is a complete, correctly separated
filament. The measured false-positive and true-positive confidence distributions
overlap substantially in this project.

This experiment adapts the quality-prediction idea from
[Mask Scoring R-CNN (Huang et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html).
It is a small second-stage CNN, not a reproduction of Mask R-CNN or its results.

## What changes

The semantic model and its checkpoint remain frozen. For each candidate mask,
`instance_quality.py` extracts a square context crop and constructs four aligned
96×96 channels: locally normalized grayscale, semantic probability, the binary
instance mask, and annotation support. Six rotation-invariant geometry and
intensity features supplement the image features. Unsupported pixels are zeroed.

Four convolution/GroupNorm/SiLU blocks feed two outputs:

- A logit for whether the mask has IoU at least 0.5 with an annotation.
- An estimate of its best ground-truth IoU.

Binary cross-entropy trains the first output; smooth-L1 trains the second.
Their sigmoid outputs are multiplied to produce the quality score. All targets
come from polygon masks on training observations, not auxiliary annotation
metadata. Augmentations transform the four channels together.

Two proposal settings are fixed in advance:

| Name | Probability threshold | Minimum area | Closing |
|---|---:|---:|---|
| Strict (current baseline) | 0.80 | 500 | Disabled |
| Permissive | 0.65 | 200 | Disabled |

The permissive setting can recover smaller or weaker regions. The quality model
can reject poor proposals, but cannot create a mask absent from both proposal
sets. It is therefore a targeted precision/recall experiment, not a claim that
PQ 0.60 is attainable.

## Reproduce

From the repository root, with the pinned dependencies and existing Round-5
checkpoint available:

```bash
python -m pytest tests/test_instance_quality.py -q
python experiment_instance_quality.py --smoke
python experiment_instance_quality.py
```

The smoke run trains for two epochs with six training observations and two
observations from each validation partition. It exercises the entire path;
its scores are not model-quality estimates. The full pilot trains for twelve
epochs on 128 randomly selected unique training observations (seed 2026).

Checkpoint selection uses the existing 16 monitoring observations. The 40
calibration observations select a proposal setting and quality cutoff from
0/0.1/0.2/0.3/0.4/0.5. A cutoff of zero retains every proposal and includes the
original baseline as a control. Settings are frozen before the 85-observation
paired assessment. A separate check requires exact reproduction of the historical
pooled baseline PQ.

The new classifier sees no validation labels during gradient updates. Its
training proposals are in-sample for the already trained U-Net; this can affect
generalization and is recorded in the protocol. The assessment set has been
reused in earlier research, so confidence intervals remain exploratory.

## Files and outputs

- [instance_quality.py](../instance_quality.py): feature extraction, CNN, filtering,
  and cached-IoU evaluation.
- [experiment_instance_quality.py](../experiment_instance_quality.py): preparation,
  training, checkpoint selection, calibration, paired evaluation, and gated export.
- `runs/instance_quality_20260929/`: source/checkpoint signatures, cached features,
  saved models, optimizer state, and measurement JSONs.
- [Experiment report](../reports/instance_quality_20260929.md): live progress and
  measured results once the full experiment starts.

Production defaults in `solution.py` are unchanged. A full-run candidate is
promoted only if pooled PQ improves and the paired mean-PQ improvement has a
positive lower 95% observation-bootstrap bound. A promoted candidate gets a
new CSV and manifest under the experiment's own `submission/` directory;
older submissions remain intact. No automatic Kaggle leaderboard upload occurs.

The current portable Kaggle notebook packages the original semantic pipeline.
It does not silently include this experimental second stage. Reproducing a
promoted second-stage export requires both checkpoints and this experiment's
frozen configuration; the manifest identifies them.
