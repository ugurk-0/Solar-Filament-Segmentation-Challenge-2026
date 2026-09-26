"""Generate the presentation notebook with nbformat; behavior lives in Python modules."""
from pathlib import Path
import shutil
import nbformat as nbf


def main():
    backup = Path("runs/pq_research_20260926/notebook_before.ipynb")
    if not backup.exists():
        shutil.copyfile("notebook.ipynb", backup)
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text.strip()))
    md("""# Solar filament segmentation: PQ experiment

This notebook presents the implementation in `solution.py` and the paired experiment in
`pq_experiment.py`. It displays the completed run by default. Enable the explicit run switches
to reproduce expensive stages in a **new output directory**.

**Hypothesis:** EDT watershed fragments elongated filaments. Connected components should
improve RQ; correcting crop sampling, clDice and annotation support should improve training.
Checkpoint selection uses 16 fixed monitoring observations, calibration uses 40 different
observations, and the paired assessment uses the remaining 85. All belong to the original
physical-observation fold 1. This is an exploratory local assessment, not a leaderboard score.

The new support mask changes the scoring protocol. The historical mean PQ 0.075559 is
context only; compare old and new checkpoints re-scored under identical settings below.
""")
    code('''
import sys, json, importlib
from pathlib import Path
import pandas as pd
import torch
from IPython.display import display, Image, FileLink
import solution
import pq_experiment
importlib.reload(solution)
importlib.reload(pq_experiment)
assert Path("solution.py").exists(), "Start the kernel in the repository root"
RUN_ROOT = Path("runs/pq_research_20260926")
CFG = pq_experiment.config(RUN_ROOT / "training", fold=1)
CFG.epochs = 4
CFG.lr = 1e-4
CFG.ema_decay = 0.9
CFG.eval_every = 1
CFG.eval_max_images = 16
CFG.init_ckpt = "runs/corrected_loss_fold1/fold1.pt"
RUN_TRAINING = False
RUN_CALIBRATION = False
RUN_ASSESSMENT = False
RUN_SUBMISSION = False
print("Python:", sys.executable, "Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Model selection: mean PQ on fixed monitor, no TTA")
print("Annotation support fraction:", solution.build_limb_mask(CFG).mean())
''')
    md(r"""## Data and mathematical checks

The projected support is $|x-c_x|\leq\sin(70^\circ)\sqrt{R^2-(y-c_y)^2}$ inside the disk
(fixed radius, zero solar tilt approximation). The same support masks losses, predictions
and validation labels. No spine, chirality or other auxiliary metadata is used for training.

PQ matches instances greedily at IoU $\geq0.5$. Mean image PQ and pooled dataset PQ are
different estimands; both are reported. Empty/empty image PQ is defined as zero here.
""")
    code('''
data_audit = RUN_ROOT / "data_audit.json"
if data_audit.exists():
    audit = json.loads(data_audit.read_text())
    display(pd.Series({k: v for k, v in audit.items() if k != "image_ids"}))
else:
    print("Run: python audit_data.py")
''')
    md("""## Training with real validation and instance previews

This run warm-starts the **same-fold** 60-epoch EMA checkpoint, with a fresh optimizer,
four additional full-data epochs, corrected mask losses and crop sampling. It is not a
from-scratch ablation of individual fixes. EMA decay is 0.9 for this short fine-tuning run.
`fold1_best.pt` is selected by measured monitor PQ; `fold1.pt` is the final epoch.
""")
    code('''
def on_epoch(epoch, checkpoint):
    print(f"Completed epoch {epoch}; displaying actual validation artifacts")
    display(Image(filename=str(Path(CFG.work_dir) / "learning_curves.png")))
    for path in sorted((Path(CFG.work_dir) / "previews").glob(f"epoch{epoch:03d}_*.png")):
        display(Image(filename=str(path)))

if RUN_TRAINING:
    solution.train(CFG, epoch_callback=on_epoch)
else:
    print("Displaying saved run. Set RUN_TRAINING=True and use a new work_dir to train again.")
history_path = Path(CFG.work_dir) / "history.json"
if history_path.exists():
    history = json.loads(history_path.read_text())
    display(pd.DataFrame(history)[["epoch", "loss", "pq", "lr", "elapsed_seconds"]])
    on_epoch(history[-1]["epoch"], None)
    display(FileLink(str(Path(CFG.work_dir) / "dashboard.html")))
''')
    md("""## Post-processing calibration and paired assessment

The 12 predeclared candidates are 2 instance methods × 3 thresholds × 2 minimum areas.
Only the 40 calibration observations select post-processing. The selected parameters are
then frozen for both checkpoints. Paired bootstrap intervals resample observations; a
date-block sensitivity interval accounts for within-day dependence. Neither captures all
temporal dependence, training-seed variability, or leaderboard uncertainty.
""")
    code('''
baseline_dir = RUN_ROOT / "baseline"
comparison_dir = RUN_ROOT / "comparison"
if RUN_CALIBRATION:
    pq_experiment.run(CFG.init_ckpt, baseline_dir, "baseline")
if RUN_ASSESSMENT:
    pq_experiment.run(Path(CFG.work_dir) / "fold1_best.pt", comparison_dir, "compare",
                      baseline_dir / "audit.json", baseline_dir / "selected_postproc.json")
grid_path = baseline_dir / "calibration_grid.json"
if grid_path.exists():
    grid = json.loads(grid_path.read_text())
    display(pd.DataFrame([{**r["parameters"], **r["summary"]} for r in grid]).sort_values("mean_pq", ascending=False))
for directory in (baseline_dir, comparison_dir):
    path = directory / "audit.json"
    if path.exists():
        report = json.loads(path.read_text())
        print(directory.name)
        display({k: v for k, v in report.items() if not k.endswith("rows")})
''')
    md("""## Full-fold descriptive metrics

These include monitoring and calibration observations, so they are descriptive rather
than an independent estimate after model selection. Visuals use fixed observation IDs.
""")
    code('''
for directory in (baseline_dir, comparison_dir):
    path = directory / "full_fold.json"
    if path.exists():
        report = json.loads(path.read_text())
        print(directory.name)
        display(pd.Series(report["summary"]))
        display(pd.DataFrame(report["rows"])[["image_id", "pq", "sq", "rq", "tp", "fp", "fn", "pixel_dice"]].describe())
        for preview in sorted((directory / "previews").glob("*.png")):
            display(Image(filename=str(preview)))
''')
    md("""## Reproducible test inference

Test labels are never used for tuning. The CSV contains one compressed COCO RLE per
instance; every RLE is decoded and checked before writing. Generating this file does not
upload it to Kaggle. Review `submission_manifest.json` for image coverage and settings.
""")
    code('''
from dataclasses import replace
submission_dir = RUN_ROOT / "submission"
if RUN_SUBMISSION:
    parameters = json.loads((baseline_dir / "selected_postproc.json").read_text())["parameters"]
    submission_cfg = replace(CFG, work_dir=str(submission_dir),
                             test_dir="MAGFiLO_1.0_Kaggle_2026/test/test_images", **parameters)
    solution.submit(submission_cfg, [str(Path(CFG.work_dir) / "fold1_best.pt")])
for name in ("submission.csv", "submission_manifest.json"):
    path = submission_dir / name
    if path.exists(): display(FileLink(str(path)))
''')
    md("""## Reproduction and limitations

Install `requirements.txt` in the selected Python environment. Run
`python -m pytest tests/test_solution_regressions.py -q`, then the smoke command documented
in `reports/pq_experiment_20260926.md`. Full commands and measured results live in that report.

Exact duplicate grouping prevents annotator leakage, but does not eliminate temporal
correlation of evolving solar structures. The annotation-region model is approximate.
The experiment uses one fold and one fine-tuning seed; no claim of challenge completion or
winning performance follows from it. See `reports/skills_review.md` for the skills assessment.
""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}})
    nbf.validate(notebook)
    nbf.write(notebook, "notebook.ipynb")


if __name__ == "__main__":
    main()
