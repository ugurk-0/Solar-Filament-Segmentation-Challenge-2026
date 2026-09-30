# Radial-correction matched fine-tuning pilot

Hypothesis: bounded illumination correction improves limb filament detection. 128 training observations, two epochs per arm, same parent, crop schedule, seed, loss, support and post-processing. This pilot does not establish the optimum training budget. A 0.005 monitoring gain over both parent and raw control is required before assessing the corrected model on 85 reused research observations. No test labels are used.

Reproduce: `python experiment_solar_preprocessing.py --smoke`, then `python experiment_solar_preprocessing.py`. Corrected checkpoints require SolarDataset or radial_correct before inference; the ordinary CLI does not apply that transform.

```json
{
  "status": "training",
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
    ]
  },
  "active_arm": "radial_corrected"
}
```
