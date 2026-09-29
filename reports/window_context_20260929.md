# Inference context experiment — 29 September 2026

Target: PQ 0.60; not an achieved score.

Hypothesis: matching the training crop context can improve inference with GroupNorm. The checkpoint, physical-observation split, support mask and no-TTA setting are fixed. Window/overlap pairs are 1024/256, 768/256, and 512/128. Thresholds are 0.65/0.80/0.90; minimum areas are 200/500. Selection uses mean PQ on the 40 calibration observations. The frozen winner is compared on 85 reused research observations; this is exploratory, not an untouched or leaderboard estimate.

Command: `python experiment_window_context.py`

Status: **calibrating**

| Window | Overlap | Threshold | Minimum area | Calibration mean PQ | Calibration pooled PQ |
|---:|---:|---:|---:|---:|---:|
