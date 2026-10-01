# Observation-balanced and spine-supervised learning

Both arms use all 566 training observations, one sampled real annotation set per visit, three epochs, same Round-5 initialization/seed/crops/loss/post-processing. The auxiliary arm adds a 0.1-weight balanced MSE on annotated spine heatmaps. One distance transform per image; support/foreground clipping; aligned augmentations. A faster learning rate is used for the previously unused auxiliary head in both arms. No annotation-derived input features are required at inference.

Run `python experiment_spine_learning.py --smoke`, then `python experiment_spine_learning.py`. Checkpoint/arm selection uses 16 monitor observations. Only the selected arm may advance to the reused 85-observation assessment. No test labels are used.

```json
{
  "status": "training",
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
      }
    ]
  },
  "active_arm": "spine_auxiliary"
}
```
