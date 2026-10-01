# Observation-balanced and spine-supervised learning

Both arms use all 566 training observations, one sampled real annotation set per visit, three epochs, same Round-5 initialization/seed/crops/loss/post-processing. The auxiliary arm adds a 0.1-weight balanced MSE on annotated spine heatmaps. One distance transform per image; support/foreground clipping; aligned augmentations. A faster learning rate is used for the previously unused auxiliary head in both arms. No annotation-derived input features are required at inference.

Run `python experiment_spine_learning.py --smoke`, then `python experiment_spine_learning.py`. Checkpoint/arm selection uses 16 monitor observations. Only the selected arm may advance to the reused 85-observation assessment. No test labels are used.

```json
{
  "status": "training",
  "arms": {},
  "active_arm": "observation_control"
}
```
