# Final submission and cleanup ? 8 November 2026

Selected: **YOLO11n-seg, epoch 17**, confidence 0.50, minimum area 128, input size 1536, no TTA. It is the retained model under the predeclared promotion criteria; the later fine-tune had slightly higher pooled PQ but lower mean PQ and did not pass the paired gate.

Local assessment: mean PQ **0.319854**, pooled PQ **0.331939**, on 85 reused observations. Kaggle submission **58139775** completed on 8 November with API-verified **public score 0.28**. Previous submission 56607737 has API-verified public score **0.23** (earlier user estimate was 0.24).

The CSV has **1,036 non-overlapping instances** over **180 processed images**, native 2048 ? 2048 COCO RLE masks. Every mask, ID, test-image filename, round trip and CSV write/read check passed. SHA-256: `cb33be0a74b270c3a14c77f686159bbd88beb8896f4e801a16f8afe842fe404e`.

Local file: `submissions/final_20261108/submission.csv`. Exact checkpoint hash and inference/source/image manifests are alongside it. Kaggle status is recorded in `reports/final_submission_status.json`.

## Cleanup decision

The user requested removal of unfruitful methods. Executable implementations and their method-only tests were removed for rejected fine-tuning rounds, medoid selection, window sweeps, quality filtering, seeded growth, radial/CLAHE preprocessing, spine supervision, the unpromoted embedding prototype, the mislabeled medium runner, and YOLO/U-Net refinement. They remain available in Git history. Reports and generated run evidence are retained so unsuccessful work is not erased from the scientific record.

Shared cached-IoU metrics moved unchanged from the retired quality-filter module to `instance_metrics.py`. The winning YOLO and promoted U-Net/skeleton lineage remain. Unused embedding-head hooks were removed; the ordinary U-Net segmentation/spine checkpoint schema remains intact. YOLO export verifies the evaluated inference functions separately from unrelated retired training code.

Removed source/test files:

- `run_training_round2.py`
- `run_training_round3.py`
- `run_training_round4.py`
- `autoupgrade.py`
- `experiment_annotation_medoid.py`
- `tests/test_annotation_medoid.py`
- `experiment_window_context.py`
- `experiment_instance_quality.py`
- `instance_quality.py`
- `tests/test_instance_quality.py`
- `experiment_seeded_instances.py`
- `seeded_instances.py`
- `tests/test_seeded_instances.py`
- `experiment_solar_preprocessing.py`
- `solar_preprocessing.py`
- `tests/test_solar_preprocessing.py`
- `experiment_spine_learning.py`
- `spine_learning.py`
- `tests/test_spine_learning.py`
- `experiment_clahe.py`
- `clahe_preprocessing.py`
- `tests/test_clahe_preprocessing.py`
- `experiment_instance_embeddings.py`
- `instance_embeddings.py`
- `tests/test_instance_embeddings.py`
- `tests/test_experiment_instance_embeddings.py`
- `experiment_yolo11m.py`
- `experiment_yolo_refinement.py`
- `yolo_refinement.py`
- `tests/test_yolo_refinement.py`

## Verification

- Full retained regression suite: `python -m pytest tests -q` ? **40 passed** in the YOLO environment. The lower count reflects removal of tests specific to retired methods, not suppressed failures.
- Post-cleanup inference helpers are checked against their evaluated AST fingerprints; a two-image export is compared exactly against the corresponding submitted rows. The accepted full CSV retains its all-image validation manifest.
- The executed notebook presents the final model, epoch curves, previews, and submission status. Historical reports remain linked as evidence.
- The two-image post-cleanup export matched all three corresponding submitted instance rows exactly. A two-epoch, eight-image U-Net smoke run completed with finite loss and validation (PQ 0.0, an execution check only). No additional training was used for the Kaggle submission.

## Reproduce

Use Python 3.12 and install `requirements-yolo.txt`. The selected checkpoint and competition images are local artifacts and are not bundled in Git.

```powershell
python submit_yolo.py --output submissions/reproduced
```

Do not overwrite the accepted submission. Use a new output directory. No new tuning or training was used to select this submission.
