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
        },
        "lr": 5e-06,
        "sq": {
          "mean": 0.6138073286530193,
          "median": 0.6251090515426512,
          "p05": 0.4508643689296481,
          "p95": 0.757375731334381
        },
        "rq": {
          "mean": 0.5276836019767709,
          "median": 0.5357142857142857,
          "p05": 0.1875,
          "p95": 0.8500000000000001
        },
        "pixel_dice": {
          "mean": 0.6437205302094737,
          "median": 0.6975562134760023,
          "p05": 0.342950759989078,
          "p95": 0.8167901340298891
        },
        "pixel_iou": {
          "mean": 0.5022533417703337,
          "median": 0.5355879682963006,
          "p05": 0.22230088495575223,
          "p95": 0.693941798941799
        },
        "tp": {
          "mean": 4.3125,
          "median": 4.0,
          "p05": 0.75,
          "p95": 8.5
        },
        "fp": {
          "mean": 4.4375,
          "median": 3.5,
          "p05": 0.0,
          "p95": 10.0
        },
        "fn": {
          "mean": 3.125,
          "median": 2.0,
          "p05": 0.75,
          "p95": 7.25
        },
        "one_to_many_gt": {
          "mean": 0.75,
          "median": 1.0,
          "p05": 0.0,
          "p95": 1.5
        },
        "many_to_one_pred": {
          "mean": 0.0625,
          "median": 0.0,
          "p05": 0.0,
          "p95": 0.25
        },
        "mean_instances_dice": {
          "mean": 0.7759399279437077,
          "median": 0.7824697557635243,
          "p05": 0.679402485400572,
          "p95": 0.8604919000287462
        },
        "mean_instances_iou": {
          "mean": 0.6378039048700047,
          "median": 0.6426696662917135,
          "p05": 0.5144909629366499,
          "p95": 0.7551696355605942
        },
        "images_per_second": 1.8859507138804255,
        "dataset_pq": 0.339833740818767,
        "dataset_sq": 0.6378039048700047,
        "dataset_rq": 0.5328185328185329,
        "total_tp": 69,
        "total_fp": 71,
        "total_fn": 50
      }
    ]
  },
  "active_arm": "radial_corrected"
}
```
