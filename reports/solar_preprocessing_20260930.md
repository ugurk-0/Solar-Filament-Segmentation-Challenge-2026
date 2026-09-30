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
        },
        "lr": 5e-06,
        "sq": {
          "mean": 0.5734144937492559,
          "median": 0.6163207307394296,
          "p05": 0.0,
          "p95": 0.7660368081965114
        },
        "rq": {
          "mean": 0.4819490913322343,
          "median": 0.5108695652173914,
          "p05": 0.0,
          "p95": 0.9166666666666666
        },
        "pixel_dice": {
          "mean": 0.6368582466167851,
          "median": 0.6888352416956619,
          "p05": 0.32632596966303573,
          "p95": 0.830998786467431
        },
        "pixel_iou": {
          "mean": 0.4932630607165405,
          "median": 0.5253723142145061,
          "p05": 0.20852849336455892,
          "p95": 0.7141767903332433
        },
        "tp": {
          "mean": 3.9375,
          "median": 5.0,
          "p05": 0.0,
          "p95": 7.5
        },
        "fp": {
          "mean": 4.6875,
          "median": 4.0,
          "p05": 0.0,
          "p95": 10.0
        },
        "fn": {
          "mean": 3.5,
          "median": 2.5,
          "p05": 0.75,
          "p95": 8.25
        },
        "one_to_many_gt": {
          "mean": 0.9375,
          "median": 1.0,
          "p05": 0.0,
          "p95": 2.25
        },
        "many_to_one_pred": {
          "mean": 0.0625,
          "median": 0.0,
          "p05": 0.0,
          "p95": 0.25
        },
        "mean_instances_dice": {
          "mean": 0.7709650894594628,
          "median": 0.7649615513206286,
          "p05": 0.680715708620349,
          "p95": 0.8608531508313437
        },
        "mean_instances_iou": {
          "mean": 0.631336285194044,
          "median": 0.6193827828911749,
          "p05": 0.515976895931042,
          "p95": 0.7557072990273688
        },
        "images_per_second": 1.6926347158463597,
        "dataset_pq": 0.3095267390445507,
        "dataset_sq": 0.631336285194044,
        "dataset_rq": 0.490272373540856,
        "total_tp": 63,
        "total_fp": 75,
        "total_fn": 56
      }
    ]
  },
  "active_arm": "raw_control"
}
```
