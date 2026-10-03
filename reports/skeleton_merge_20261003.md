# Skeleton-endpoint graph merge (Qiuwei V2 reproduction)

Post-processing only: no retraining, no test data. Merge parameters tuned on the 40 calibration observations; comparison on the 85 reused audit observations against the identical cached probabilities with components-only decoding.

```json
{
  "status": "complete",
  "grid": [
    {
      "max_distance": 25,
      "min_cosine": 0.7,
      "min_gap_probability": 0.3
    },
    {
      "max_distance": 25,
      "min_cosine": 0.7,
      "min_gap_probability": 0.4
    },
    {
      "max_distance": 25,
      "min_cosine": 0.8,
      "min_gap_probability": 0.3
    },
    {
      "max_distance": 25,
      "min_cosine": 0.8,
      "min_gap_probability": 0.4
    },
    {
      "max_distance": 50,
      "min_cosine": 0.7,
      "min_gap_probability": 0.3
    },
    {
      "max_distance": 50,
      "min_cosine": 0.7,
      "min_gap_probability": 0.4
    },
    {
      "max_distance": 50,
      "min_cosine": 0.8,
      "min_gap_probability": 0.3
    },
    {
      "max_distance": 50,
      "min_cosine": 0.8,
      "min_gap_probability": 0.4
    }
  ],
  "selection": {
    "parameters": {
      "max_distance": 50,
      "min_cosine": 0.8,
      "min_gap_probability": 0.3
    },
    "summary": {
      "n": 40,
      "mean_pq": 0.28528726511346714,
      "dataset_pq": 0.2734979465497837,
      "sq": 0.6342323070852743,
      "rq": 0.4312267657992565,
      "tp": 116,
      "fp": 153,
      "fn": 153,
      "pixel_dice": 0.5998730264065243
    }
  },
  "assessment": {
    "baseline": {
      "n": 85,
      "mean_pq": 0.271428171269003,
      "dataset_pq": 0.27100326957602094,
      "sq": 0.6427512490316036,
      "rq": 0.4216300940438871,
      "tp": 269,
      "fp": 359,
      "fn": 379,
      "pixel_dice": 0.6049729058047526
    },
    "merged": {
      "n": 85,
      "mean_pq": 0.28912812459228787,
      "dataset_pq": 0.28776214576331116,
      "sq": 0.642049948880506,
      "rq": 0.44819277108433736,
      "tp": 279,
      "fp": 318,
      "fn": 369,
      "pixel_dice": 0.6049729058047526
    },
    "paired": {
      "mean_pq_delta": 0.017699953323284938,
      "observation_bootstrap_95ci": [
        0.0061218126508023,
        0.031016975865807626
      ],
      "date_block_bootstrap_95ci": [
        0.006250007499515074,
        0.030685719489256646
      ],
      "n_observations": 85,
      "n_dates": 84,
      "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
    },
    "parameters": {
      "max_distance": 50,
      "min_cosine": 0.8,
      "min_gap_probability": 0.3
    }
  },
  "promoted": true,
  "decision": "promoted_research_candidate"
}
```
