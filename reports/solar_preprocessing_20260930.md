# Radial-correction matched fine-tuning pilot

Hypothesis: bounded illumination correction improves limb filament detection. 128 training observations, two epochs per arm, same parent, crop schedule, seed, loss, support and post-processing. This pilot does not establish the optimum training budget. A 0.005 monitoring gain over both parent and raw control is required before assessing the corrected model on 85 reused research observations. No test labels are used.

Reproduce: `python experiment_solar_preprocessing.py --smoke`, then `python experiment_solar_preprocessing.py`. Corrected checkpoints require SolarDataset or radial_correct before inference; the ordinary CLI does not apply that transform.

```json
{
  "status": "training",
  "arms": {},
  "active_arm": "raw_control"
}
```
