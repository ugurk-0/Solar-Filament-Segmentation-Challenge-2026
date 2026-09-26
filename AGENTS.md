# Agent Guidance

## Repository map

- The primary Kaggle solution is the root [README.md](README.md), [solution.py](solution.py), and [notebook.ipynb](notebook.ipynb).
- `MAGFiLO_1.0_Kaggle_2026/` contains the competition train/test data. Treat annotation JSON and image filenames as data contracts; do not invent labels or silently change paths.
- `stage-sohaib-fmi-detector/` is a separate nested FMI detector project with its own `src/`, `configs/`, `scripts/`, `tests/`, and checkpoints. Do not apply root-pipeline assumptions to it.
- `main.tex` is the technical report and must agree with measured experiments, not intended results.

## Reproducibility and commands

From the repository root, install the pinned dependencies before running Python or notebook code:

```powershell
python -m pip install -r requirements.txt
python solution.py train --fold 0 --epochs 2 --limit 8
python solution.py eval --fold 0
python solution.py tune --fold 0
python solution.py submit --ckpts runs/fold0.pt runs/fold1.pt
```

Use `--data-dir`, `--test-dir`, and `--work-dir` for local paths; the defaults target Kaggle paths. Smoke tests cap epochs when `--limit` is set and write checkpoints/post-processing/submissions under the configured work directory. Never overwrite valuable runs accidentally.

The notebook is a presentation layer over `solution.py`, not a second implementation. Execute cells in dependency order after selecting the intended Python kernel. If an import fails, report the interpreter and missing package, then repair the environment or code at the owning boundary; do not hide the failure by changing `sys.path` blindly.

## Mathematical and data invariants

- Group by the physical observation key `Path(file_name).stem.split("-")[-1]`; never split duplicate annotator records by COCO image id. Validation must deduplicate to one annotation set per physical observation.
- The target metric is panoptic quality, not pixel accuracy: for IoU threshold $\tau=0.5$, greedily match unmatched prediction/ground-truth instances with IoU $\geq\tau$, then compute
  $\mathrm{PQ}=\mathrm{SQ}\,\mathrm{RQ}$,
  $\mathrm{SQ}=|TP|^{-1}\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)$,
  and $\mathrm{RQ}=TP/(TP+FP/2+FN/2)$. Preserve instance identity through post-processing and test `panoptic_quality` on empty, one-to-one, merged, split, and below-threshold cases.
- Apply the annotated-region limb mask consistently to training losses and inference. Do not evaluate or tune outside the annotation support.
- Keep image, polygon mask, spine-distance target, valid-region mask, and every geometric augmentation in the same coordinate system. `spine` points are `(x, y)` while array indices are `(row=y, col=x)`.
- Compare experiments using the same physical-observation split, seed, checkpoint semantics, TTA, and post-processing. Do not tune post-processing on the final test set or report unmeasured scores.

## Change and validation protocol

- Start at the function that computes or mutates the behavior; keep edits narrow and preserve public names unless a migration is required.
- For model or metric changes, first run the smallest focused check, then the smoke training command, then the relevant local evaluation. Inspect shapes, dtypes, finite values, and mask coverage at boundaries.
- For notebook changes, refresh the notebook summary, validate the edited cell in the selected kernel, and confirm that imports resolve from the repository root. Do not edit notebook JSON by hand when a notebook-aware operation is available.
- Do not claim success from import-only validation when the changed behavior is training, inference, post-processing, or scoring.
- Preserve user changes and generated artifacts. Avoid committing checkpoints, datasets, submissions, or large notebook outputs unless explicitly requested.

## Reporting

Every substantive change should state the hypothesis, files changed, command(s) run, and observed result. Separate measured values from expectations. Link to existing documentation instead of copying it into agent guidance.