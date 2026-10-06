# YOLO-guided U-Net boundary refinement

No additional training: intersect each dilated YOLO instance with retained U-Net probabilities, then resolve overlap by YOLO confidence. Eight predeclared settings are calibrated on 40 observations. Only a calibration winner over direct YOLO advances to the reused 85-observation assessment. No test labels used.

```json
{
  "status": "calibrating",
  "grid": [
    {
      "radius": 8,
      "threshold": 0.5,
      "confidence": 0.1,
      "min_area": 128
    },
    {
      "radius": 8,
      "threshold": 0.5,
      "confidence": 0.3,
      "min_area": 128
    },
    {
      "radius": 8,
      "threshold": 0.8,
      "confidence": 0.1,
      "min_area": 128
    },
    {
      "radius": 8,
      "threshold": 0.8,
      "confidence": 0.3,
      "min_area": 128
    },
    {
      "radius": 24,
      "threshold": 0.5,
      "confidence": 0.1,
      "min_area": 128
    },
    {
      "radius": 24,
      "threshold": 0.5,
      "confidence": 0.3,
      "min_area": 128
    },
    {
      "radius": 24,
      "threshold": 0.8,
      "confidence": 0.1,
      "min_area": 128
    },
    {
      "radius": 24,
      "threshold": 0.8,
      "confidence": 0.3,
      "min_area": 128
    }
  ]
}
```
