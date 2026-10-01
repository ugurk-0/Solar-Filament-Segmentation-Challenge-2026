# Observation-balanced and spine-supervised learning

Both arms use all 566 training observations, one sampled real annotation set per visit, three epochs, same Round-5 initialization/seed/crops/loss/post-processing. The auxiliary arm adds a 0.1-weight balanced MSE on annotated spine heatmaps. One distance transform per image; support/foreground clipping; aligned augmentations. A faster learning rate is used for the previously unused auxiliary head in both arms. No annotation-derived input features are required at inference.

Run `python experiment_spine_learning.py --smoke`, then `python experiment_spine_learning.py`. Checkpoint/arm selection uses 16 monitor observations. Only the selected arm may advance to the reused 85-observation assessment. No test labels are used.

```json
{
  "status": "complete",
  "arms": {
    "observation_control": [
      {
        "epoch": 1,
        "mean_loss": 0.26371800955450786,
        "segmentation_loss": 0.26371800955450786,
        "spine_loss": 0.0,
        "monitor_pq": 0.32171449024807386,
        "seconds": 211.4220000000205
      },
      {
        "epoch": 2,
        "mean_loss": 0.26806032979472577,
        "segmentation_loss": 0.26806032979472577,
        "spine_loss": 0.0,
        "monitor_pq": 0.32095691968162343,
        "seconds": 205.92199999996228
      },
      {
        "epoch": 3,
        "mean_loss": 0.26934265596101015,
        "segmentation_loss": 0.26934265596101015,
        "spine_loss": 0.0,
        "monitor_pq": 0.3155607584058949,
        "seconds": 190.64100000000326
      }
    ],
    "spine_auxiliary": [
      {
        "epoch": 1,
        "mean_loss": 0.2737346515191107,
        "segmentation_loss": 0.26372313994400914,
        "spine_loss": 0.10011511565228864,
        "monitor_pq": 0.32289983001064393,
        "seconds": 469.64100000000326
      },
      {
        "epoch": 2,
        "mean_loss": 0.27594234649579014,
        "segmentation_loss": 0.26806966185095876,
        "spine_loss": 0.07872684452807788,
        "monitor_pq": 0.32205260028762217,
        "seconds": 405.875
      },
      {
        "epoch": 3,
        "mean_loss": 0.27673534884417017,
        "segmentation_loss": 0.2693302291692251,
        "spine_loss": 0.07405119632574805,
        "monitor_pq": 0.3172414252685115,
        "seconds": 397.65600000001723
      }
    ]
  },
  "active_arm": "spine_auxiliary",
  "parent_monitor_pq": 0.31377332193790564,
  "best_monitor": {
    "observation_control": 0.32171449024807386,
    "spine_auxiliary": 0.32289983001064393
  },
  "selected": "spine_auxiliary",
  "decision": "rejected_assessment",
  "promoted": false,
  "assessment": {
    "summary": {
      "n": 85,
      "mean_pq": 0.2678590180472862,
      "dataset_pq": 0.26800523184274033,
      "sq": 0.640731026498107,
      "rq": 0.418280402788536,
      "tp": 270,
      "fp": 373,
      "fn": 378
    },
    "paired": {
      "mean_pq_delta": -0.003569153221716838,
      "observation_bootstrap_95ci": [
        -0.007764917204432637,
        -4.508300700872966e-05
      ],
      "date_block_bootstrap_95ci": [
        -0.0078089855050171315,
        6.179037009103074e-05
      ],
      "n_observations": 85,
      "n_dates": 84,
      "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
    }
  },
  "leaderboard_score": null
}
```
