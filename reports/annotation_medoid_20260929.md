# Annotation-medoid experiment — 29 September 2026

One actual representative annotation per observation reduces contradictory supervision and duplicate weighting.

Two-annotator groups tie; deterministic selection cannot identify the correct annotator. This combines target selection and observation reweighting.

Fixed parent, fold, no TTA, support mask, and post-processing. Six epochs; checkpoint selected on 16 monitor observations. The 85-observation assessment is reused research data. No test labels are used.

Reproduce: `python experiment_annotation_medoid.py --smoke`, then `python experiment_annotation_medoid.py`.

```json
{
  "status": "training",
  "target_pq": 0.6,
  "configuration": {
    "data_dir": "C:\\Users\\ugurk\\Documents\\kaggle\\filament-segmentation\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train",
    "train_json": "C:\\Users\\ugurk\\Documents\\kaggle\\filament-segmentation\\filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    "test_dir": "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images",
    "work_dir": "runs\\annotation_medoid_20260929\\training",
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
    "epochs": 6,
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
  "selected_training_observations": 566,
  "leaderboard_score": null,
  "completed_epochs": 4,
  "history": [
    {
      "epoch": 1,
      "loss": 0.2668023730098359,
      "pq": 0.3124388090153526,
      "lr": 9.330127018922195e-06
    },
    {
      "epoch": 2,
      "loss": 0.2645237886679257,
      "pq": 0.3203640582308928,
      "lr": 7.500000000000002e-06
    },
    {
      "epoch": 3,
      "loss": 0.2702514309396592,
      "pq": 0.32030933229866415,
      "lr": 5.000000000000001e-06
    },
    {
      "epoch": 4,
      "loss": 0.2707525842589435,
      "pq": 0.32437515685903073,
      "lr": 2.500000000000002e-06
    }
  ]
}
```
