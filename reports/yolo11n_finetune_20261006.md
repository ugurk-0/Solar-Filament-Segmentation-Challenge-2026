# YOLO11n follow-up experiment

Actual loaded model: {'task': 'segment', 'scale': 'n', 'parameters': 2842803, 'yaml_file': 'yolo11n-seg.yaml'}. Initial checkpoint: `runs\yolo11n_20261003\best_pq.pt`. Input 1536, batch 2, nominal batch 8, learning rate 0.00015, budget 12 epochs. Reset optimizer for fine-tuning. Same grouped split, support-aware loss, no geometric augmentation or TTA. The parent checkpoint is retained unless monitoring PQ improves. Calibrate on 40 observations, then compare on the reused 85-observation assessment against `yolo11n_20261003` (mean 0.3199, pooled 0.3319). Promotion requires higher pooled PQ and paired mean-PQ confidence interval above zero.

```json
{
  "status": "calibrating",
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
    },
    {
      "epoch": 5,
      "monitor": {
        "n": 16,
        "mean_pq": 0.353097673157451,
        "dataset_pq": 0.32348743768919297,
        "sq": 0.6402355537598611,
        "rq": 0.5052631578947369,
        "tp": 72,
        "fp": 94,
        "fn": 47
      },
      "elapsed_seconds": 580.375
    },
    {
      "epoch": 6,
      "monitor": {
        "n": 16,
        "mean_pq": 0.3384172780331041,
        "dataset_pq": 0.31548752651912154,
        "sq": 0.6469780435138507,
        "rq": 0.4876325088339223,
        "tp": 69,
        "fp": 95,
        "fn": 50
      },
      "elapsed_seconds": 693.0470000000205
    },
    {
      "epoch": 7,
      "monitor": {
        "n": 16,
        "mean_pq": 0.33786080447065475,
        "dataset_pq": 0.3034371064029463,
        "sq": 0.6420553265917414,
        "rq": 0.4726027397260274,
        "tp": 69,
        "fp": 104,
        "fn": 50
      },
      "elapsed_seconds": 807.7660000000615
    },
    {
      "epoch": 8,
      "monitor": {
        "n": 16,
        "mean_pq": 0.3403080618721471,
        "dataset_pq": 0.3171821913984827,
        "sq": 0.6432990924138241,
        "rq": 0.4930555555555556,
        "tp": 71,
        "fp": 98,
        "fn": 48
      },
      "elapsed_seconds": 921.875
    },
    {
      "epoch": 9,
      "monitor": {
        "n": 16,
        "mean_pq": 0.3383845366544298,
        "dataset_pq": 0.31426408238568704,
        "sq": 0.6418069288158398,
        "rq": 0.4896551724137931,
        "tp": 71,
        "fp": 100,
        "fn": 48
      },
      "elapsed_seconds": 1031.734999999986
    },
    {
      "epoch": 10,
      "monitor": {
        "n": 16,
        "mean_pq": 0.31838040528806144,
        "dataset_pq": 0.2946212585880912,
        "sq": 0.6472739772011094,
        "rq": 0.45517241379310347,
        "tp": 66,
        "fp": 105,
        "fn": 53
      },
      "elapsed_seconds": 1141.6720000000205
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
  "best_epoch": 5,
  "training_complete": true,
  "recovered_error": "MemoryError: Unable to allocate 4.00 MiB for an array with shape (2048, 2048) and data type bool"
}
```
