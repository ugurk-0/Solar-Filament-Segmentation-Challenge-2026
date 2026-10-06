# YOLO11n follow-up experiment

Actual loaded model: {'task': 'segment', 'scale': 'n', 'parameters': 2842803, 'yaml_file': 'yolo11n-seg.yaml'}. Initial checkpoint: `runs\yolo11n_20261003\best_pq.pt`. Input 1536, batch 2, nominal batch 8, learning rate 0.00015, budget 12 epochs. Reset optimizer for fine-tuning. Same grouped split, support-aware loss, no geometric augmentation or TTA. The parent checkpoint is retained unless monitoring PQ improves. Calibrate on 40 observations, then compare on the reused 85-observation assessment against the retained nano candidate (mean 0.3199, pooled 0.3319). Promotion requires higher pooled PQ and paired mean-PQ confidence interval above zero.

```json
{
  "status": "training",
  "history": [
    {
      "epoch": 1,
      "monitor": {
        "n": 16,
        "mean_pq": 0.2951834359840594,
        "dataset_pq": 0.27633337366176663,
        "sq": 0.6495896470407201,
        "rq": 0.4253968253968254,
        "tp": 67,
        "fp": 129,
        "fn": 52
      },
      "elapsed_seconds": 119.85999999998603
    },
    {
      "epoch": 2,
      "monitor": {
        "n": 16,
        "mean_pq": 0.3006861525946317,
        "dataset_pq": 0.2858853410434353,
        "sq": 0.6411999791974192,
        "rq": 0.445859872611465,
        "tp": 70,
        "fp": 125,
        "fn": 49
      },
      "elapsed_seconds": 245.204000000027
    },
    {
      "epoch": 3,
      "monitor": {
        "n": 16,
        "mean_pq": 0.3092039567064997,
        "dataset_pq": 0.2870753603483421,
        "sq": 0.6469907375014874,
        "rq": 0.44370860927152317,
        "tp": 67,
        "fp": 116,
        "fn": 52
      },
      "elapsed_seconds": 356.625
    },
    {
      "epoch": 4,
      "monitor": {
        "n": 16,
        "mean_pq": 0.3518203110626821,
        "dataset_pq": 0.30253370669860674,
        "sq": 0.6411908410627187,
        "rq": 0.47183098591549294,
        "tp": 67,
        "fp": 98,
        "fn": 52
      },
      "elapsed_seconds": 467.954000000027
    }
  ],
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
  "best_epoch": 4
}
```
