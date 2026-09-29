# Inference context experiment — 29 September 2026

Target: PQ 0.60; not an achieved score.

Hypothesis: matching the training crop context can improve inference with GroupNorm. The checkpoint, physical-observation split, support mask and no-TTA setting are fixed. Window/overlap pairs are 1024/256, 768/256, and 512/128. Thresholds are 0.65/0.80/0.90; minimum areas are 200/500. Selection uses mean PQ on the 40 calibration observations. The frozen winner is compared on 85 reused research observations; this is exploratory, not an untouched or leaderboard estimate.

Command: `python experiment_window_context.py`

Status: **assessing_frozen_winner**

| Window | Overlap | Threshold | Minimum area | Calibration mean PQ | Calibration pooled PQ |
|---:|---:|---:|---:|---:|---:|
| 1024 | 256 | 0.65 | 200 | 0.228916 | 0.229882 |
| 1024 | 256 | 0.65 | 500 | 0.255454 | 0.248211 |
| 1024 | 256 | 0.8 | 200 | 0.241730 | 0.242679 |
| 1024 | 256 | 0.8 | 500 | 0.269978 | 0.258613 |
| 1024 | 256 | 0.9 | 200 | 0.243063 | 0.248111 |
| 1024 | 256 | 0.9 | 500 | 0.262993 | 0.258206 |
| 768 | 256 | 0.65 | 200 | 0.221099 | 0.233346 |
| 768 | 256 | 0.65 | 500 | 0.232901 | 0.244907 |
| 768 | 256 | 0.8 | 200 | 0.230328 | 0.235697 |
| 768 | 256 | 0.8 | 500 | 0.253350 | 0.249743 |
| 768 | 256 | 0.9 | 200 | 0.233187 | 0.237085 |
| 768 | 256 | 0.9 | 500 | 0.247986 | 0.242499 |
| 512 | 128 | 0.65 | 200 | 0.219942 | 0.230448 |
| 512 | 128 | 0.65 | 500 | 0.242868 | 0.249132 |
| 512 | 128 | 0.8 | 200 | 0.228583 | 0.240405 |
| 512 | 128 | 0.8 | 500 | 0.249757 | 0.252604 |
| 512 | 128 | 0.9 | 200 | 0.231811 | 0.234371 |
| 512 | 128 | 0.9 | 500 | 0.253464 | 0.244517 |
