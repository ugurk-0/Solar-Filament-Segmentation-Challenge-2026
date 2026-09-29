# Instance-quality CNN experiment

Hypothesis: predicted instance quality separates false detections better than mean pixel probability. A permissive proposal set can also retain smaller objects. Frozen Round-5 U-Net, same fold/support/no-TTA. 128 training observations; 16 monitor, 40 calibration, 85 reused assessment observations. No test labels or external pretrained weights are used.

Inspired by [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html); this is a separate lightweight adaptation, not that architecture or its reported results.

Run: `python experiment_instance_quality.py --smoke`, then `python experiment_instance_quality.py`.

```json
{
  "status": "complete",
  "history": [
    {
      "epoch": 1,
      "loss": 0.7151724471455126,
      "monitor_pq": 0.3137796382843129,
      "seconds": 1.2339999999967404
    },
    {
      "epoch": 2,
      "loss": 0.7084181512351585,
      "monitor_pq": 0.3137796382843129,
      "seconds": 0.6570000000065193
    },
    {
      "epoch": 3,
      "loss": 0.7037030226361435,
      "monitor_pq": 0.3137796382843129,
      "seconds": 0.7649999999994179
    },
    {
      "epoch": 4,
      "loss": 0.6874010078674924,
      "monitor_pq": 0.3198560135619928,
      "seconds": 0.7029999999940628
    },
    {
      "epoch": 5,
      "loss": 0.6643671836473245,
      "monitor_pq": 0.321263117163769,
      "seconds": 0.7960000000020955
    },
    {
      "epoch": 6,
      "loss": 0.6305822711075302,
      "monitor_pq": 0.3137796382843129,
      "seconds": 0.7809999999881256
    },
    {
      "epoch": 7,
      "loss": 0.5843194350732112,
      "monitor_pq": 0.3137796382843129,
      "seconds": 0.9839999999967404
    },
    {
      "epoch": 8,
      "loss": 0.5608017483643726,
      "monitor_pq": 0.3173365658655146,
      "seconds": 0.8130000000091968
    },
    {
      "epoch": 9,
      "loss": 0.5525640497165444,
      "monitor_pq": 0.3137796382843129,
      "seconds": 0.7969999999913853
    },
    {
      "epoch": 10,
      "loss": 0.5427509528345766,
      "monitor_pq": 0.31615329224680144,
      "seconds": 0.8440000000118744
    },
    {
      "epoch": 11,
      "loss": 0.5347703955869759,
      "monitor_pq": 0.31972255029265595,
      "seconds": 0.9530000000086147
    },
    {
      "epoch": 12,
      "loss": 0.5367632228716285,
      "monitor_pq": 0.3137796382843129,
      "seconds": 0.7810000000026776
    }
  ],
  "smoke": false,
  "prepared_images": 85,
  "training_proposals": 2260,
  "positive_proposals": 912,
  "selected": {
    "proposal": "strict",
    "cutoff": 0.2
  },
  "calibration": [
    {
      "proposal": "strict",
      "cutoff": 0.0,
      "summary": {
        "n": 40,
        "mean_pq": 0.26997800891733376,
        "dataset_pq": 0.2586128473444123,
        "sq": 0.6340551827435372,
        "rq": 0.407871198568873,
        "tp": 114,
        "fp": 176,
        "fn": 155
      }
    },
    {
      "proposal": "strict",
      "cutoff": 0.1,
      "summary": {
        "n": 40,
        "mean_pq": 0.27371772519240467,
        "dataset_pq": 0.2633234638716329,
        "sq": 0.6340551827435372,
        "rq": 0.41530054644808745,
        "tp": 114,
        "fp": 166,
        "fn": 155
      }
    },
    {
      "proposal": "strict",
      "cutoff": 0.2,
      "summary": {
        "n": 40,
        "mean_pq": 0.28674830558900977,
        "dataset_pq": 0.2755799165737574,
        "sq": 0.6365896072853796,
        "rq": 0.4329004329004329,
        "tp": 100,
        "fp": 93,
        "fn": 169
      }
    },
    {
      "proposal": "strict",
      "cutoff": 0.3,
      "summary": {
        "n": 40,
        "mean_pq": 0.029668349250297175,
        "dataset_pq": 0.03513540481422448,
        "sq": 0.7001984245120451,
        "rq": 0.05017921146953405,
        "tp": 7,
        "fp": 3,
        "fn": 262
      }
    },
    {
      "proposal": "strict",
      "cutoff": 0.4,
      "summary": {
        "n": 40,
        "mean_pq": 0.0,
        "dataset_pq": 0.0,
        "sq": 0.0,
        "rq": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 269
      }
    },
    {
      "proposal": "strict",
      "cutoff": 0.5,
      "summary": {
        "n": 40,
        "mean_pq": 0.0,
        "dataset_pq": 0.0,
        "sq": 0.0,
        "rq": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 269
      }
    },
    {
      "proposal": "permissive",
      "cutoff": 0.0,
      "summary": {
        "n": 40,
        "mean_pq": 0.2289161462749613,
        "dataset_pq": 0.2298817123673769,
        "sq": 0.6379217518194709,
        "rq": 0.36036036036036034,
        "tp": 120,
        "fp": 277,
        "fn": 149
      }
    },
    {
      "proposal": "permissive",
      "cutoff": 0.1,
      "summary": {
        "n": 40,
        "mean_pq": 0.23251337134597794,
        "dataset_pq": 0.2344582242521792,
        "sq": 0.6379217518194709,
        "rq": 0.3675344563552833,
        "tp": 120,
        "fp": 264,
        "fn": 149
      }
    },
    {
      "proposal": "permissive",
      "cutoff": 0.2,
      "summary": {
        "n": 40,
        "mean_pq": 0.25078165255830054,
        "dataset_pq": 0.2562555162500907,
        "sq": 0.6429683862275003,
        "rq": 0.39855072463768115,
        "tp": 110,
        "fp": 173,
        "fn": 159
      }
    },
    {
      "proposal": "permissive",
      "cutoff": 0.3,
      "summary": {
        "n": 40,
        "mean_pq": 0.039936253388853496,
        "dataset_pq": 0.049327973310191744,
        "sq": 0.6979908223392132,
        "rq": 0.0706713780918728,
        "tp": 10,
        "fp": 4,
        "fn": 259
      }
    },
    {
      "proposal": "permissive",
      "cutoff": 0.4,
      "summary": {
        "n": 40,
        "mean_pq": 0.0,
        "dataset_pq": 0.0,
        "sq": 0.0,
        "rq": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 269
      }
    },
    {
      "proposal": "permissive",
      "cutoff": 0.5,
      "summary": {
        "n": 40,
        "mean_pq": 0.0,
        "dataset_pq": 0.0,
        "sq": 0.0,
        "rq": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 269
      }
    }
  ],
  "assessment": {
    "baseline": {
      "n": 85,
      "mean_pq": 0.271428171269003,
      "dataset_pq": 0.27100326957602094,
      "sq": 0.6427512490316036,
      "rq": 0.4216300940438871,
      "tp": 269,
      "fp": 359,
      "fn": 379
    },
    "candidate": {
      "n": 85,
      "mean_pq": 0.2618123638515019,
      "dataset_pq": 0.2694015361290879,
      "sq": 0.6451457838880789,
      "rq": 0.4175824175824176,
      "tp": 228,
      "fp": 216,
      "fn": 420
    },
    "paired": {
      "mean_pq_delta": -0.009615807417501109,
      "observation_bootstrap_95ci": [
        -0.029611273948861427,
        0.007415759539944124
      ],
      "date_block_bootstrap_95ci": [
        -0.029651708951635034,
        0.0076475772677867815
      ],
      "n_observations": 85,
      "n_dates": 84,
      "interpretation": "Conditional on this split, training seed and selected model; not leaderboard uncertainty."
    }
  },
  "promoted": false,
  "leaderboard_score": null
}
```
