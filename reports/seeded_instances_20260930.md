# Seeded mask-growth experiment

Hypothesis: growing confident components into weaker connected foreground improves shapes without admitting unseeded false detections. Compare merging versus preserving confident seeds. The frozen Round-5 model, no-TTA inference, observation split and support mask are unchanged.

Seven settings including the baseline were compared on 40 calibration observations. The selected setting was frozen before evaluating 85 reused research observations. Repeated assessment is exploratory; these are not leaderboard scores.

Run `python experiment_seeded_instances.py --smoke`, then `python experiment_seeded_instances.py`. No neural-network weights are retrained. The retained baseline and submissions are preserved.

```json
{
  "baseline": {
    "n": 85,
    "mean_pq": 0.271428171269003,
    "dataset_pq": 0.27100326957602094,
    "sq": 0.6427512490316036,
    "rq": 0.4216300940438871,
    "tp": 269,
    "fp": 359,
    "fn": 379
  },
  "candidate": {
    "n": 85,
    "mean_pq": 0.271428171269003,
    "dataset_pq": 0.27100326957602094,
    "sq": 0.6427512490316036,
    "rq": 0.4216300940438871,
    "tp": 269,
    "fp": 359,
    "fn": 379
  },
  "paired": {
    "mean_pq_delta": 0.0,
    "observation_bootstrap_95ci": [
      0.0,
      0.0
    ],
    "date_block_bootstrap_95ci": [
      0.0,
      0.0
    ],
    "n_observations": 85,
    "n_dates": 84,
    "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
  },
  "promoted": false,
  "selected": null,
  "smoke": false,
  "leaderboard_score": null
}
```
