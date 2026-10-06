# YOLO-guided U-Net boundary refinement

No additional training: intersect each dilated YOLO instance with retained U-Net probabilities, then resolve overlap by YOLO confidence. Eight predeclared settings are calibrated on 40 observations. Only a calibration winner over direct YOLO advances to the reused 85-observation assessment. No test labels used.

```json
{
  "status": "complete",
  "grid": [
    {
      "parameters": {
        "radius": 8,
        "threshold": 0.5,
        "confidence": 0.1,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.27356946341973265,
        "dataset_pq": 0.2793350311413763,
        "sq": 0.6219569052757207,
        "rq": 0.44912280701754387,
        "tp": 128,
        "fp": 173,
        "fn": 141
      }
    },
    {
      "parameters": {
        "radius": 8,
        "threshold": 0.5,
        "confidence": 0.3,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.27878526003747134,
        "dataset_pq": 0.27780963810485465,
        "sq": 0.6275086123859656,
        "rq": 0.44271844660194176,
        "tp": 114,
        "fp": 132,
        "fn": 155
      }
    },
    {
      "parameters": {
        "radius": 8,
        "threshold": 0.8,
        "confidence": 0.1,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.2838937460131315,
        "dataset_pq": 0.28291354640438887,
        "sq": 0.6259740672412069,
        "rq": 0.45195729537366547,
        "tp": 127,
        "fp": 166,
        "fn": 142
      }
    },
    {
      "parameters": {
        "radius": 8,
        "threshold": 0.8,
        "confidence": 0.3,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.28647559939667916,
        "dataset_pq": 0.2803755625631349,
        "sq": 0.6314653156842286,
        "rq": 0.444007858546169,
        "tp": 113,
        "fp": 127,
        "fn": 156
      }
    },
    {
      "parameters": {
        "radius": 24,
        "threshold": 0.5,
        "confidence": 0.1,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.26784749987722767,
        "dataset_pq": 0.2742345804566367,
        "sq": 0.624157905119305,
        "rq": 0.43936731107205623,
        "tp": 125,
        "fp": 175,
        "fn": 144
      }
    },
    {
      "parameters": {
        "radius": 24,
        "threshold": 0.5,
        "confidence": 0.3,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.2677280611015004,
        "dataset_pq": 0.2679688570464935,
        "sq": 0.6318164794582461,
        "rq": 0.42412451361867703,
        "tp": 109,
        "fp": 136,
        "fn": 160
      }
    },
    {
      "parameters": {
        "radius": 24,
        "threshold": 0.8,
        "confidence": 0.1,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.28485393282821964,
        "dataset_pq": 0.28279636939088815,
        "sq": 0.6257148015656658,
        "rq": 0.45195729537366547,
        "tp": 127,
        "fp": 166,
        "fn": 142
      }
    },
    {
      "parameters": {
        "radius": 24,
        "threshold": 0.8,
        "confidence": 0.3,
        "min_area": 128
      },
      "summary": {
        "n": 40,
        "mean_pq": 0.28239017842012826,
        "dataset_pq": 0.27581264292093655,
        "sq": 0.6323812398502554,
        "rq": 0.4361493123772102,
        "tp": 111,
        "fp": 129,
        "fn": 158
      }
    }
  ],
  "selected": {
    "parameters": {
      "radius": 8,
      "threshold": 0.8,
      "confidence": 0.3,
      "min_area": 128
    },
    "summary": {
      "n": 40,
      "mean_pq": 0.28647559939667916,
      "dataset_pq": 0.2803755625631349,
      "sq": 0.6314653156842286,
      "rq": 0.444007858546169,
      "tp": 113,
      "fp": 127,
      "fn": 156
    }
  },
  "direct_yolo_calibration": {
    "n": 40,
    "mean_pq": 0.28536580957114055,
    "dataset_pq": 0.2889436716700023,
    "sq": 0.6403616507281132,
    "rq": 0.45121951219512196,
    "tp": 111,
    "fp": 112,
    "fn": 158
  },
  "calibration_passed": false,
  "promoted": false,
  "decision": "rejected_calibration",
  "leaderboard_score": null
}
```
