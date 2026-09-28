# Post-submission improvement experiment

The project author's first Kaggle result is **0.24** (user-reported). The initial CSV uses Round 5 EMA weights, eight-way TTA, connected components, probability threshold 0.65 and minimum area 200.

## Reference and hypothesis

Reviewed [Fususu's topology-safe handoff](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff), version 2, downloaded on 27 September 2026. It attributes the underlying detector/refiner architecture to [Nomannic19](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner).

The reference uses a Mask R-CNN detector followed by a crop U-Net, rather than our full-image semantic model. It filters detector confidence, refines crops, and accepts flip-averaged predictions only when area, overlap and connectivity checks pass. Those checks operate on individual refiner crops and cannot be assumed effective on our full-disk probability maps without testing.

Its stated 0.70 score dates to 5 August. The [competition announcement](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements) says the leaderboard was rescored on 12 August after adopting PQ. Do not compare the old 0.70 directly with the current 0.24 or promise to reproduce it under the current evaluator.

No reference code or checkpoints are incorporated into the production model in this experiment. The first hypothesis is that stronger confidence/area filtering reduces false positives, while removing morphological closing may preserve separation between nearby filaments.

## Protocol

`refine_postprocessing.py` evaluates 16 predeclared combinations: probability thresholds 0.50/0.65/0.80/0.90, minimum areas 200/500, and closing sizes 1/3. It uses the existing 40-observation calibration partition only for selection. The chosen combination is frozen before the 85-observation paired assessment. The checkpoint is unchanged, and cached probabilities are checked against its SHA-256 and inference configuration. All passes use no TTA, matching the historical assessment.

The 85-observation assessment has been reused in earlier research, so improvements remain exploratory. A new Kaggle score can only be established by submission, not inferred from this local comparison.

Commands and artifacts:

```powershell
python refine_postprocessing.py
```

- Live log: `runs/refinement_20260927.log`
- Protocol, calibration grid and paired assessment: `runs/pq_refinement_20260927/`
- Notebook: post-submission refinement section in `notebook.ipynb`

## Evaluator comparison

The [organizer's self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) pools TP/FP/FN across observations and uses IoU strictly greater than 0.5. It counts qualifying overlaps directly; our documented local metric uses greedy one-to-one matching at IoU at least 0.5. We retain the established local metric for paired comparability and report pooled dataset PQ as well as mean image PQ. Neither is claimed to be an exact reproduction of all Kaggle server behavior.

## Observed result

Calibration selected threshold **0.80**, minimum area **500**, and closing size **1** (disabled). Its calibration mean PQ was 0.2698. Frozen-setting assessment:

| Configuration | Mean image PQ | Dataset PQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Original Round 5 post-processing | 0.2632 | 0.2626 | 305 | 536 | 343 |
| Calibrated candidate | 0.2714 | 0.2710 | 269 | 359 | 379 |

The paired mean-PQ change is **+0.0082**, with observation-bootstrap 95% interval **[-0.0096, +0.0262]**. This is a promising candidate, not a statistically established gain. It removes 177 false positives while losing 36 true detections. The original 0.24-scoring CSV is preserved.

Export uses **no TTA**, matching this local assessment, rather than changing inference settings after evaluation:

```powershell
python solution.py submit --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images --work-dir runs/pq_refinement_20260927 --ckpts runs/pq_research_20260926/training_round5/fold1_best.pt --no-tta --window 1024
```

Regression check: `python -m pytest tests -q` — 15 passed. This experiment changes settings only; it does not change training, the model, or the metric implementation. Neither the external detector/refiner nor topology-safe TTA was implemented or validated as a new model here.

## 28 September correction and continuation

The candidate export **failed** its non-overlap check at `20171021223430Lh.jpeg_4`; no completed CSV was written. Instance-wise small-hole filling could absorb a separate nested component. The converter now preserves the original component when filling would collide with another instance. Two focused regression tests cover nested ownership and ordinary hole filling; the full suite passes 17 tests.

Calibration and assessment are rerun under `runs/pq_refinement_20260928_fixed/` with the corrected converter. Both baseline and candidate are rescored, rather than comparing against historical cleanup behavior. The automatic training batch and all keep/reject decisions are documented in [the upgrade report](autoupgrade_20260928.md).

The corrected assessment completed with unchanged mean PQ 0.2714 and dataset PQ 0.2710. Export then completed successfully: **1,328 instances, all 180 test images processed, no overlaps, all RLE round trips passed**, no TTA. The candidate is `runs/pq_refinement_20260928_fixed/submission.csv`; the original scored CSV remains untouched. This candidate has not been submitted to Kaggle.

## Original training notebook review

Downloaded version 1 of the attributed Mask R-CNN + U-Net training notebook. Its validation manifest groups physical observations, but its **final detector uses the entire manifest** and its final refiner rebuilds the crop cache from all observations. Those final checkpoints are therefore unsuitable for an unbiased comparison on our held-out training-data fold. The architecture can be retrained on our split; importing the final weights and reporting their performance on that same fold would not establish generalization.

The refiner uses pretrained ResNet-18 features and equally weighted BCE/Dice on 256-pixel instance-centered crops. The detector is a pretrained Mask R-CNN ResNet-50 FPN v2. These are substantive architectural differences from our full-image semantic model; the public 0.70 title cannot be attributed to TTA alone. A future detector/refiner experiment must retain our physical-observation split and record training provenance.

## Organizer-counting audit

Ran `python audit_official_metric.py` on the candidate's cached probabilities and the same 85 deduplicated, support-masked assessment observations. Strict-IoU, direct-overlap counting from the organizer notebook produced **0.2710032696 pooled PQ**, with TP 269, FP 359 and FN 379. No observation had different TP/FP/FN counts from our local greedy rule. Thus the rule difference does not explain this candidate's local score. This does not verify hidden-server annotation selection or performance on the test set. Machine-readable evidence: [official-counting audit](official_counting_audit_20260928.json).

## Instance-error diagnosis

Ran `python analyze_instance_errors.py` using the same assessment predictions. Of 379 missed instances, 106 are under 500 pixels, 130 are 500–1499 pixels and 143 are at least 1500 pixels. Thus missed small objects are important but do not explain all failures. Among missed instances, 144 have best predicted IoU in [0.3, 0.5], 53 in [0.1, 0.3), and 182 below 0.1. Better delineation could recover some near-threshold matches, but that hypothesis still requires validation.

False-positive mean probability has median 0.9687 versus 0.9753 for true positives; the distributions overlap substantially. Raising an instance-confidence threshold is therefore not an obvious route to a large improvement. The current reduced-clDice experiment tests a mask-shape hypothesis; a later crop-refiner or detector experiment should use the same physical split rather than importing final weights trained on all observations. Evidence: [error diagnosis](instance_errors_20260928.json). No new hyperparameters were selected from these assessment diagnostics during the running batch.
