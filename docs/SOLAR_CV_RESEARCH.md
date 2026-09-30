# Methods for noisy, projected solar-filament images

Research and implementation review: 30 September 2026. The retained local mean
PQ is 0.2714; paper results on Dice, accuracy, or another dataset are not directly
comparable to this competition's instance PQ.

## What the literature supports

| Problem | Relevant primary material | Assessment for this repository |
|---|---|---|
| Limb darkening and projected geometry | [Joshi et al., automated filament detection](https://arxiv.org/abs/0905.3055) corrects limb darkening and foreshortening. | Test illumination correction first, preserving original image coordinates. Geometric remapping requires consistent mask/inverse-projection handling. |
| Observatory and illumination variation | [Diercke et al., 2024](https://arxiv.org/html/2402.15407v1) combines detection and segmentation across solar observatories and describes disk normalization and instrument-specific corrections. | Detector-guided refinement is relevant, but a two-stage system adds missed-detection risk. Separate preprocessing experiments from architecture changes. |
| GONG image standardization | [New dataset and framework for filament detection, 2026](https://www.sciencedirect.com/science/article/pii/S0094576526004649) uses radial-profile correction and disk-only histogram standardization. | Annular medians are a directly relevant starting point. The implemented pilot uses bounded partial correction, not the paper's complete histogram-matching pipeline. |
| Noise without clean reference frames | [Noise2Void](https://arxiv.org/html/1811.10980v1) learns from individual noisy images, assuming conditionally pixel-independent noise. | JPEG blocks and atmospheric structures may violate that assumption. This is an inference about suitability, not a measured failure here. First test realistic mild corruption augmentation; inspect whether faint filament features survive any denoising. |
| Spherical geometry | [Spherical CNNs](https://arxiv.org/abs/1801.10130) builds rotation-equivariant operations for signals on a sphere. | Our inputs are projected disk JPEGs, not fully sampled spherical signals. A spherical network is not a drop-in replacement. A projected viewing-angle channel or local surface patches is a smaller next experiment. |
| Thin structures and limited compute | [Flat U-Net](https://arxiv.org/html/2502.07259v1) uses lightweight channel-attention blocks for solar filaments. | Relevant alternative backbone, but its semantic Dice/recall results do not establish an instance-PQ gain here. |
| Separate filament instances | [CondInst](https://arxiv.org/abs/2003.05664) predicts instance masks with conditional convolutions. | A stronger architectural alternative to connected components when merges/splits dominate. It requires a new training/evaluation path, not a post-processing switch. |

## What “on a sphere” changes

For an approximately orthographic, centred disk, normalized projected radius
`rho = sqrt((x-cx)^2 + (y-cy)^2)/R` gives `mu = sqrt(max(0, 1-rho^2))` for a
spherical surface. Toward the limb, a surface patch is increasingly compressed
in the viewing direction. Correcting brightness does not undo this compression.
Unwarping cannot reconstruct information that the image did not resolve.

My implementation assessment is to retain labels and final masks in their native
2048-by-2048 coordinates. A future geometry-channel experiment can provide `mu`
to the network with the same crops/flips as the image. More ambitious surface
patches require calibrated disk geometry and identical image/mask coordinate
transforms. Filaments have height above the surface, which limits an exact
surface-only correction. The existing approximate annotation support remains
unchanged in this pilot.

## Implemented experiment

Six sampled training JPEGs showed median central intensity around 0.55 and
intensity around 0.30-0.32 at normalized radii 0.93-0.96. This small diagnostic
supports investigating residual limb darkening; it is not a dataset-wide estimate.

`solar_preprocessing.radial_correct` estimates an intensity profile with 64
annular medians from a subsample, smooths the profile (not the image), and applies
a bounded multiplicative gain. The gain is blended at strength 0.5. Coordinates,
polygon masks and annotation support are unchanged; exterior pixels retain their
values. The algorithm does not denoise, deproject, or apply histogram matching.

The matched pilot trains two copies of the existing Round-5 U-Net:

- Raw-image control, with no correction.
- Identical fine-tuning with radial correction in both training and evaluation.

Both use the same 128 unique training observations, two epochs, seed, crop
schedule, loss, learning rate, checkpoint initialization, and post-processing.
The 16-observation monitor selects checkpoints. Correction must beat the parent
and raw control by 0.005 before assessment on 85 reused research observations.
This short pilot tests compatibility, not the best possible normalization model.
No test images or labels select settings.

```bash
python -m pytest tests/test_solar_preprocessing.py -q
python experiment_solar_preprocessing.py --smoke
python experiment_solar_preprocessing.py
```

Files changed: [solar_preprocessing.py](../solar_preprocessing.py),
[experiment_solar_preprocessing.py](../experiment_solar_preprocessing.py),
[focused tests](../tests/test_solar_preprocessing.py), and this research note.
The experiment uses a process-local dataset subclass, leaving production
`solution.py` unchanged. Corrected checkpoints require that subclass or explicit
`radial_correct(image, radius=950, strength=0.5)` before `predict_full`;
the normal CLI and portable notebook do not automatically apply the transform.
Checkpoints record `radial_strength` and are isolated under
`runs/solar_preprocessing_20260930/`.

The smoke run completed both training arms and evaluation. All 35 tests passed
before the full pilot. [Live experiment report](../reports/solar_preprocessing_20260930.md).
