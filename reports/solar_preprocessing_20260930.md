# Radial-correction matched fine-tuning pilot

Hypothesis: bounded illumination correction improves limb filament detection. 128 training observations, two epochs per arm, same parent, crop schedule, seed, loss, support and post-processing. This pilot does not establish the optimum training budget. A 0.005 monitoring gain over both parent and raw control is required before assessing the corrected model on 85 reused research observations. No test labels are used.

Reproduce: `python experiment_solar_preprocessing.py --smoke`, then `python experiment_solar_preprocessing.py`. Corrected checkpoints require SolarDataset or radial_correct before inference; the ordinary CLI does not apply that transform.

```json
{
  "status": "complete",
  "arms": {
    "raw_control": [
      {
        "epoch": 1,
        "loss": 0.26865239220205694,
        "pq": 0.3176469248222315,
        "elapsed_seconds": {
          "mean": 0.5907949249994999,
          "median": 0.5621113999950467,
          "p05": 0.5111137250059983,
          "p95": 0.7578191499997047
        }
      },
      {
        "epoch": 2,
        "loss": 0.30097432917682454,
        "pq": 0.3183885029572583,
        "elapsed_seconds": {
          "mean": 0.5890190249974694,
          "median": 0.5876061000017216,
          "p05": 0.5537381500034826,
          "p95": 0.6287742749991594
        }
      }
    ],
    "radial_corrected": [
      {
        "epoch": 1,
        "loss": 0.2719928557635285,
        "pq": 0.35032822367905164,
        "elapsed_seconds": {
          "mean": 0.5302365500010637,
          "median": 0.5286433000001125,
          "p05": 0.5164901499883854,
          "p95": 0.5514393750054296
        }
      },
      {
        "epoch": 2,
        "loss": 0.3061113358126022,
        "pq": 0.35297060294840504,
        "elapsed_seconds": {
          "mean": 0.5696635375024925,
          "median": 0.5579563500068616,
          "p05": 0.5294026500123437,
          "p95": 0.6382197750062915
        }
      }
    ]
  },
  "active_arm": "radial_corrected",
  "parent_monitor_pq": 0.31377332193790564,
  "raw_monitor_pq": 0.3183885029572583,
  "corrected_monitor_pq": 0.35297060294840504,
  "decision": "rejected_assessment",
  "promoted": false,
  "assessment": {
    "summary": {
      "n": 85,
      "mean_pq": 0.2820779065382112,
      "dataset_pq": 0.2707426771333483,
      "sq": 0.6440977923973074,
      "rq": 0.42034405385190726,
      "tp": 281,
      "fp": 408,
      "fn": 367
    },
    "paired": {
      "mean_pq_delta": 0.01064973526920815,
      "observation_bootstrap_95ci": [
        -0.008471030024939133,
        0.03293694683861468
      ],
      "date_block_bootstrap_95ci": [
        -0.008076719255326189,
        0.03392266260702144
      ],
      "n_observations": 85,
      "n_dates": 84,
      "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
    }
  },
  "leaderboard_score": null
}
```
