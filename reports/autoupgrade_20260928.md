# Automatic improvement batch — 28 September 2026

User-reported Kaggle baseline: **0.24**. No automatic leaderboard claim.

Each run starts from Round 5, using the fixed observation split and corrected post-processing. Epochs are selected on 16 monitor observations. Only candidates exceeding the starting monitor PQ by 0.005 are assessed on the reused 85-observation research split. Promotion requires gains in both mean and pooled PQ and a positive lower paired-bootstrap bound. This conservative gate does not remove selection bias from repeated research. Test labels are never read.

Progress and epoch previews: `runs/autoupgrade_20260928/<experiment>/`. Reproduce with `python autoupgrade.py`. Completed runs are preserved.

## Export defect found before this batch

The previous candidate export stopped on overlapping instances at 20171021223430Lh.jpeg. Small-hole filling could absorb a nested component. Cleanup now falls back to the original component whenever filled pixels would collide with another instance. Two synthetic regression tests cover nested ownership and ordinary hole filling. Calibration and assessment are rerun with the corrected converter before this batch.

Starting monitor PQ: 0.31377332193790564

Corrected candidate export: complete

## less_cldice

Reducing skeleton-loss weight may recover complete masks and improve instance IoU.

Status: **rejected_assessment**

```json
{
  "hypothesis": "Reducing skeleton-loss weight may recover complete masks and improve instance IoU.",
  "status": "rejected_assessment",
  "config": {
    "data_dir": "C:\\Users\\ugurk\\Documents\\kaggle\\filament-segmentation\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train",
    "train_json": "C:\\Users\\ugurk\\Documents\\kaggle\\filament-segmentation\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    "test_dir": "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    "work_dir": "runs\\autoupgrade_20260928\\less_cldice",
    "progress_log": "",
    "img_size": 2048,
    "crop_size": 768,
    "oversample_p": 0.5,
    "disk_radius": 950,
    "limb_max_deg": 70.0,
    "limb_geometry": "longitude",
    "encoder": "resnet18",
    "pretrained": false,
    "decoder_ch": [
      256,
      128,
      64,
      32,
      16
    ],
    "bn_ch": 256,
    "bn_dilations": [
      1,
      2,
      4
    ],
    "epochs": 3,
    "eval_every": 1,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "num_workers": 0,
    "lr": 1e-05,
    "weight_decay": 0.0001,
    "amp": true,
    "ema_decay": 0.99,
    "init_ckpt": "runs\\pq_research_20260926\\training_round5\\fold1_best.pt",
    "eval_max_images": 16,
    "preview_count": 3,
    "w_bce": 0.3,
    "w_dice": 0.5,
    "w_cld": 0.2,
    "w_spine": 0.0,
    "pos_weight_cap": 1.0,
    "tta": false,
    "tta_batch_size": 2,
    "window": 1024,
    "overlap": 256,
    "thresh": 0.8,
    "min_area": 500,
    "min_dist_ws": 25,
    "close_kernel": 1,
    "instance_method": "components",
    "tune_thresh": [
      0.3,
      0.4,
      0.5,
      0.6,
      0.7
    ],
    "tune_mdist": [
      15,
      25,
      40
    ],
    "tune_marea": [
      30,
      50,
      100
    ],
    "n_folds": 5,
    "fold": 1,
    "seed": 2026,
    "limit": 0,
    "device": "cuda"
  },
  "completed_epochs": 3,
  "history": [
    {
      "epoch": 1,
      "loss": 0.2891176807676984,
      "pq": 0.322392957671921,
      "lr": 7.500000000000001e-06
    },
    {
      "epoch": 2,
      "loss": 0.29437226306481584,
      "pq": 0.31870880169110793,
      "lr": 2.5000000000000015e-06
    },
    {
      "epoch": 3,
      "loss": 0.30238548232765133,
      "pq": 0.31490112689459143,
      "lr": 0.0
    }
  ],
  "best_monitor_pq": 0.322392957671921,
  "assessment": {
    "n": 85,
    "mean_pq": 0.27355599039022604,
    "dataset_pq": 0.2714173321694195,
    "sq": 0.6438732332486228,
    "rq": 0.42153846153846153,
    "tp": 274,
    "fp": 378,
    "fn": 374,
    "pixel_dice": 0.6102484560815009
  },
  "paired_vs_parent": {
    "mean_pq_delta": 0.0021278191212230207,
    "observation_bootstrap_95ci": [
      -0.005388184627857457,
      0.011592124814881748
    ],
    "date_block_bootstrap_95ci": [
      -0.005153721760629347,
      0.011516371872147815
    ],
    "n_observations": 85,
    "n_dates": 84,
    "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
  }
}
```

## more_random_crops

Reducing filament-centered sampling to 25% may suppress remaining false detections.

Status: **rejected_assessment**

```json
{
  "hypothesis": "Reducing filament-centered sampling to 25% may suppress remaining false detections.",
  "status": "rejected_assessment",
  "config": {
    "data_dir": "C:\\Users\\ugurk\\Documents\\kaggle\\filament-segmentation\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train",
    "train_json": "C:\\Users\\ugurk\\Documents\\kaggle\\filament-segmentation\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    "test_dir": "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    "work_dir": "runs\\autoupgrade_20260928\\more_random_crops",
    "progress_log": "",
    "img_size": 2048,
    "crop_size": 768,
    "oversample_p": 0.25,
    "disk_radius": 950,
    "limb_max_deg": 70.0,
    "limb_geometry": "longitude",
    "encoder": "resnet18",
    "pretrained": false,
    "decoder_ch": [
      256,
      128,
      64,
      32,
      16
    ],
    "bn_ch": 256,
    "bn_dilations": [
      1,
      2,
      4
    ],
    "epochs": 3,
    "eval_every": 1,
    "batch_size": 1,
    "grad_accum_steps": 8,
    "num_workers": 0,
    "lr": 1e-05,
    "weight_decay": 0.0001,
    "amp": true,
    "ema_decay": 0.99,
    "init_ckpt": "runs\\pq_research_20260926\\training_round5\\fold1_best.pt",
    "eval_max_images": 16,
    "preview_count": 3,
    "w_bce": 0.3,
    "w_dice": 0.3,
    "w_cld": 0.4,
    "w_spine": 0.0,
    "pos_weight_cap": 1.0,
    "tta": false,
    "tta_batch_size": 2,
    "window": 1024,
    "overlap": 256,
    "thresh": 0.8,
    "min_area": 500,
    "min_dist_ws": 25,
    "close_kernel": 1,
    "instance_method": "components",
    "tune_thresh": [
      0.3,
      0.4,
      0.5,
      0.6,
      0.7
    ],
    "tune_mdist": [
      15,
      25,
      40
    ],
    "tune_marea": [
      30,
      50,
      100
    ],
    "n_folds": 5,
    "fold": 1,
    "seed": 2026,
    "limit": 0,
    "device": "cuda"
  },
  "completed_epochs": 3,
  "history": [
    {
      "epoch": 1,
      "loss": 0.30229340832305107,
      "pq": 0.31788390876516026,
      "lr": 7.500000000000001e-06
    },
    {
      "epoch": 2,
      "loss": 0.3061078032145715,
      "pq": 0.32824920295462257,
      "lr": 2.5000000000000015e-06
    },
    {
      "epoch": 3,
      "loss": 0.2893276565939285,
      "pq": 0.32178571789334864,
      "lr": 0.0
    }
  ],
  "best_monitor_pq": 0.32824920295462257,
  "assessment": {
    "n": 85,
    "mean_pq": 0.2701833685090404,
    "dataset_pq": 0.26499615917157027,
    "sq": 0.6423385987523804,
    "rq": 0.4125490196078431,
    "tp": 263,
    "fp": 364,
    "fn": 385,
    "pixel_dice": 0.6080817273299912
  },
  "paired_vs_parent": {
    "mean_pq_delta": -0.0012448027599625423,
    "observation_bootstrap_95ci": [
      -0.01068287277364067,
      0.009526566850533822
    ],
    "date_block_bootstrap_95ci": [
      -0.010415391152183703,
      0.009265954949480707
    ],
    "n_observations": 85,
    "n_dates": 84,
    "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
  }
}
```
