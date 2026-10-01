# Learning from the available annotations

Audit and controlled experiment: 1 October 2026.

## Measured inventory

The local competition training JSON contains **1,154 annotation records for 707
physical observations**, with **8,199 filament annotations**. Of the observations,
411 have one annotation set, 145 have two, and 151 have three. Grouping uses the
physical filename stem, not the COCO annotation-record ID.

The fixed fold has **566 training observations** (923 annotation records) and
141 deduplicated validation observations. The observation sets do not overlap.
All learning diagnostics below use only the fold's 6,586 training annotations.
[Machine-readable audit](../reports/learning_data_audit_20261001.json).

| Available information | Audit finding | Learning use |
|---|---|---|
| Polygon masks and bounding boxes | Median annotated area 1,231.5 pixels; 983 annotations are below 500 pixels | Keep instance identity for future instance-model training; investigate object-balanced crops and small-object recall separately |
| Spine polylines | Present and well formed in every training annotation; median length 111.3 pixels | Implemented as an auxiliary centre-line heatmap target |
| Repeated annotation sets | Some observations otherwise receive two or three times the sampling frequency | Implemented equal observation sampling with one real annotation set selected per visit |
| Categories | Left: 2,007; Right: 2,090; Unidentifiable: 2,489; no Ambiguous examples in the training fold | These are chirality labels, not confidence scores; not used in this trial |
| Observatory suffix | All six observed training station codes also occur in test filenames | Useful for future station-stratified diagnostics, not a target-derived input |
| Observation timestamps | Uneven coverage from 2011 through 2022 | Useful for temporal robustness analysis; do not treat nearby observations as independent evidence |

The [dataset authors](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437148/) describe
the polygon, bounding-box, spine, and magnetic-chirality annotations. We use the
actual competition JSON counts above, which differ from the full published
dataset. Spines and categories are training labels; the model cannot require
these labels as test-time inputs.

## Why the current learning leaves information unused

The production dataset unions polygons into a foreground mask. Its auxiliary
target is zero, and `w_spine=0` disables the existing auxiliary head. The new trial
uses the real spine annotation, rather than merely enabling the old loss against
the placeholder. It does not yet turn the semantic U-Net into a true instance
segmentation model.

About **96.58% of training spine vertices lie inside their own polygon** in the
point-in-polygon audit. Some boundary discrepancies therefore exist. The new target
clips rasterized centre lines and the resulting heatmap to foreground and
annotation support. Points are interpreted as `(x,y)`; arrays use `[y,x]`.
Targets undergo the same crop, rotation and flips as their images.

## Implemented comparison

Both arms initialize from the retained Round-5 checkpoint and use all 566 training
observations for three epochs. Each observation contributes once per epoch, with
one of its actual annotation sets sampled on that visit. Annotator polygons are
not merged into a synthetic consensus.

The **observation control** uses the existing segmentation objective. The
**spine auxiliary** arm adds a 0.1-weight balanced mean-squared-error objective
on a Gaussian distance heatmap around annotated centre lines (sigma 3 pixels).
Positive and background errors are normalized separately so empty pixels do not
dominate the auxiliary loss. The auxiliary target is not an input channel.

The existing backbone/decoder uses learning rate 1e-5. The previously unused
spine head uses 1e-3 in both arms; it receives no gradient in the control.
Other settings match: seed, raw images, crop distribution, segmentation losses,
gradient accumulation, cosine schedule, EMA, support and instance post-processing.
The final incomplete accumulation group uses its actual size.

For efficiency, one distance transform constructs the image's target from the
union of rasterized spines instead of a full-image distance transform for every
filament. The dataset caches image/target tuples. Compact best-EMA checkpoints
omit optimizer state and remain compatible with the standard inference model.
Completed training arms can be reused; interrupted training is preserved and
requires a new output directory rather than an approximate resume.

Checkpoint and arm selection use the existing 16 monitoring observations. The
winner advances to assessment only if its monitoring PQ exceeds the parent's by
0.005. Assessment uses the same 85 reused research observations. Promotion
requires improved pooled PQ and a positive lower paired-bootstrap bound for the
mean-PQ change. Repeated assessment remains exploratory. No test labels are used.

## Reproduce and inspect

```bash
python audit_learning_data.py
python -m pytest tests/test_spine_learning.py -q
python experiment_spine_learning.py --smoke
python experiment_spine_learning.py
```

The focused checks and both two-epoch smoke arms completed successfully. The
smoke scores are execution checks, not performance estimates.

- [Target construction and observation sampling](../spine_learning.py)
- [Training and assessment runner](../experiment_spine_learning.py)
- [Live results](../reports/spine_learning_20261001.md)
- Checkpoints, source/annotation hashes and fixed partitions:
  `runs/spine_learning_20261001/`

The existing baseline, preprocessing experiments and submissions are preserved.
The inference input and submission schema are unchanged by spine supervision.
