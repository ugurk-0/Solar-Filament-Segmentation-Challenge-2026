# YOLO11n follow-up experiment

Actual loaded model: {'task': 'segment', 'scale': 'n', 'parameters': 2842803, 'yaml_file': 'yolo11n-seg.yaml'}. Initial checkpoint: `runs\yolo11n_20261003\best_pq.pt`. Input 1536, batch 2, nominal batch 8, learning rate 0.00015, budget 12 epochs. Reset optimizer for fine-tuning. Same grouped split, support-aware loss, no geometric augmentation or TTA. The parent checkpoint is retained unless monitoring PQ improves. Calibrate on 40 observations, then compare on the reused 85-observation assessment against the retained nano candidate (mean 0.3199, pooled 0.3319). Promotion requires higher pooled PQ and paired mean-PQ confidence interval above zero.

```json
{
  "status": "training",
  "history": [],
  "initial_monitor": {
    "n": 16,
    "mean_pq": 0.3417497475249917,
    "dataset_pq": 0.3087535037160554,
    "sq": 0.646176975634316,
    "rq": 0.4778156996587031,
    "tp": 70,
    "fp": 104,
    "fn": 49
  },
  "actual_model": {
    "task": "segment",
    "scale": "n",
    "parameters": 2842803,
    "yaml_file": "yolo11n-seg.yaml"
  },
  "best_epoch": 0
}
```
