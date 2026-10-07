"""Generate the presentation notebook with nbformat; behavior lives in Python modules."""
from pathlib import Path
import shutil
import nbformat as nbf


def yolo_cells():
    """Self-contained views; no training or GPU imports in the notebook UI."""
    sources = [
        ('markdown', '''## YOLO runs and prediction images

Run these two cells from the repository root in order. Install
`requirements-notebook.txt` in the selected notebook kernel first.
The first cell shows the retained YOLO baseline, including actual cached predictions.
The second follows an active run automatically; use **Run** to compare experiments
and **Images** to inspect individual epochs. **Auto refresh** updates every 15 seconds.
Turning it off stops refreshing the display; training continues in its own process.

Each preview shows the H-alpha image, annotated instances, and predicted instances,
with both a full disk and a close-up. Colors distinguish instances within each panel;
matching colors across panels do not imply a matched pair. Images are reduced only for
display; PQ is computed with full-resolution masks. Fixed monitoring examples appear
after each epoch. Historical runs show two fixed calibration examples instead.

The curves show **monitoring PQ**, used to select a checkpoint. The assessment table
uses the reused 85-observation assessment split and is not a Kaggle leaderboard score.
Run artifacts live under `runs/`; a fresh clone has no trained models or preview images.
The later cells retain the earlier U-Net experiments for comparison.'''),
        ('code', '''from pathlib import Path
assert Path("yolo_dashboard.py").is_file(), "Start the notebook kernel in the repository root"
from yolo_dashboard import display_snapshot, dashboard
_ = display_snapshot(Path("runs/yolo11n_20261003"))'''),
        ('code', '''YOLO_PANEL = dashboard(Path("runs"), auto_refresh=True, interval=15)
# YOLO_PANEL.close()  # optional: stop this display's automatic refresh'''),
    ]
    return [getattr(nbf.v4, f'new_{kind}_cell')(source,
            metadata={'tags': [f'yolo-dashboard-{index}']})
            for index, (kind, source) in enumerate(sources)]


def refresh_yolo_view(path='notebook.ipynb'):
    """Update only dashboard cells, preserving other cells and their outputs."""
    notebook = nbf.read(path, as_version=4)
    notebook.cells = [cell for cell in notebook.cells
                      if not any(tag.startswith('yolo-dashboard-')
                                 for tag in cell.metadata.get('tags', []))]
    notebook.cells[0].source = notebook.cells[0].source.replace(
        '# Solar filament segmentation: PQ experiment',
        '# Solar filament segmentation: YOLO runs and PQ experiments')
    summary = '\n\n**Current view:** live YOLO status, monitoring curves, and instance prediction images are below. The later U-Net sections document earlier experiments.'
    if '**Current view:**' not in notebook.cells[0].source:
        notebook.cells[0].source += summary
    notebook.cells[1:1] = yolo_cells()
    nbf.validate(notebook)
    nbf.write(notebook, path)


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
from IPython.display import display, Image, FileLink, clear_output
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
LIVE_WATCH_MINUTES = 0  # set to e.g. 30 and rerun the training-status cell
print("Python:", sys.executable, "Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("Model selection: mean PQ on fixed monitor, no TTA")
print("Annotation support fraction:", solution.build_limb_mask(CFG).mean())
stage_paths = {
    "Data audit": RUN_ROOT / "data_audit.json",
    "Smoke training": RUN_ROOT / "smoke" / "history.json",
    "Round 1 training": RUN_ROOT / "training" / "history.json",
    "Calibration": RUN_ROOT / "baseline" / "selected_postproc.json",
    "Independent baseline assessment": RUN_ROOT / "baseline" / "audit.json",
    "Independent trained assessment": RUN_ROOT / "comparison" / "audit.json",
    "Round 2 guided training": RUN_ROOT / "training_round2" / "history.json",
    "Round 3 FP-focused training": RUN_ROOT / "training_round3" / "history.json",
    "Round 4 unweighted-BCE training": RUN_ROOT / "training_round4" / "history.json",
    "Round 5 background-exposure training": RUN_ROOT / "training_round5" / "history.json",
    "Test inference/submission": RUN_ROOT / "submission_round5" / "submission_manifest.json",
}
display(pd.DataFrame([
    {"stage": name, "status": "complete" if path.exists() else "pending", "artifact": str(path)}
    for name, path in stage_paths.items()
]))
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

Round 1 warm-starts the **same-fold** 60-epoch EMA checkpoint for four epochs. Round 2
starts from round 1's best checkpoint for three lower-learning-rate epochs and evaluates
every epoch with the frozen calibrated converter (`components`, threshold 0.65,
minimum area 200). Both use the same 16 physical observations for monitoring.

The cell below displays every completed epoch's learning curve and prediction overlays.
While training is active, set `LIVE_WATCH_MINUTES` above and rerun it to refresh every
30 seconds. Training runs in a separate process, so closing the notebook does not stop it.
""")
    code('''
import time

def show_training_run(run_dir, label):
    run_dir = Path(run_dir)
    print(f"{label}: {run_dir}")
    progress = run_dir / "training_progress.log"
    if progress.exists():
        print(progress.read_text())
    history_path = run_dir / "history.json"
    if not history_path.exists():
        print("No completed epoch yet.")
        return
    history = json.loads(history_path.read_text())
    display(pd.DataFrame(history)[["epoch", "loss", "pq", "lr", "elapsed_seconds"]])
    curves = run_dir / "learning_curves.png"
    if curves.exists(): display(Image(filename=str(curves)))
    for epoch in [row["epoch"] for row in history]:
        print(f"Prediction overlays: epoch {epoch}")
        for path in sorted((run_dir / "previews").glob(f"epoch{epoch:03d}_*.png")):
            display(Image(filename=str(path)))
    dashboard = run_dir / "dashboard.html"
    if dashboard.exists(): display(FileLink(str(dashboard)))

def show_all_training():
    show_training_run(RUN_ROOT / "training", "Round 1")
    show_training_run(RUN_ROOT / "training_round2", "Round 2 (live/current)")
    show_training_run(RUN_ROOT / "training_round3", "Round 3 (live/current)")
    show_training_run(RUN_ROOT / "training_round4", "Round 4 (live/current)")
    show_training_run(RUN_ROOT / "training_round5", "Round 5 (selected)")
    comparison = RUN_ROOT / "comparison" / "full_fold.json"
    split_path = RUN_ROOT / "training" / "split.json"
    if comparison.exists() and split_path.exists():
        rows = json.loads(comparison.read_text())["rows"]
        monitor = set(json.loads(split_path.read_text())["monitor_ids"])
        starting_pq = sum(r["pq"] for r in rows if r["image_id"] in monitor) / len(monitor)
        print(f"Round-2 starting checkpoint, calibrated monitor PQ: {starting_pq:.4f}")
        print("Round 5 was promoted only after a separate 85-observation assessment.")

if RUN_TRAINING:
    solution.train(CFG)
else:
    print("Displaying all saved training rounds. Dedicated launchers are run_training_round1.py through run_training_round5.py.")

deadline = time.time() + LIVE_WATCH_MINUTES * 60
while True:
    clear_output(wait=True)
    show_all_training()
    if LIVE_WATCH_MINUTES <= 0 or time.time() >= deadline:
        break
    time.sleep(30)
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
submission_dir = RUN_ROOT / "submission_round5"
if RUN_SUBMISSION:
    parameters = json.loads((baseline_dir / "selected_postproc.json").read_text())["parameters"]
    submission_cfg = replace(CFG, work_dir=str(submission_dir),
                             test_dir="MAGFiLO_1.0_Kaggle_2026/test/test_images", **parameters)
    solution.submit(submission_cfg, [str(RUN_ROOT / "training_round5" / "fold1_best.pt")])
for name in ("submission.csv", "submission_manifest.json"):
    path = submission_dir / name
    if path.exists(): display(FileLink(str(path)))
''')
    md("""## Post-submission refinement (27 September)

The first submitted CSV scored **0.24**, as reported by the author. The reference
notebook's 0.70 result dates to before the August metric change and is not a
directly comparable target. This experiment calibrates confidence, area and
closing on 40 observations, then evaluates the frozen winner on the same
85-observation research assessment. It does not tune on test images.

Run `python refine_postprocessing.py` to reproduce. Live progress is in
`runs/refinement_20260927.log`. No training is performed in this stage.
""")
    code('''
refinement = Path("runs/pq_refinement_20260928_fixed")
progress = Path("runs/refinement_20260928_fixed.log")
if progress.exists():
    encoding = "utf-16" if progress.read_bytes().startswith(b"\\xff\\xfe") else "utf-8"
    print("\\n".join(progress.read_text(encoding=encoding, errors="replace").splitlines()[-12:]))
grid = refinement / "calibration_grid.json"
if grid.exists():
    display(pd.DataFrame([{**r["parameters"], **r["summary"]}
        for r in json.loads(grid.read_text())]).sort_values("mean_pq", ascending=False))
assessment = refinement / "assessment.json"
if assessment.exists():
    result = json.loads(assessment.read_text())
    display(pd.DataFrame({"before": result["before"], "after": result["after"]}).T)
    display(result["paired"])
for name in ("submission.csv", "submission_description.md"):
    path = refinement / name
    if path.exists(): display(FileLink(str(path)))
''')
    md("""## Automatic improvement loop

Every experiment records its hypothesis, configuration, epoch results and decision
in `reports/autoupgrade_20260928.md`. Training previews below update from the same
run artifacts as the training process. Rerun this cell to see new epochs.
The earlier refinement export failed its overlap check; ownership-preserving
cleanup was corrected and evaluated before these experiments.
""")
    code('''
upgrade_root = Path("runs/autoupgrade_20260928")
state_path = upgrade_root / "state.json"
if state_path.exists():
    state = json.loads(state_path.read_text())
    print("Starting monitor PQ:", state.get("baseline_monitor_pq", "pending"))
    for name, record in state.get("experiments", {}).items():
        print(name, record["status"], record["hypothesis"])
        show_training_run(upgrade_root / name, name)
        if "paired_vs_parent" in record: display(record["paired_vs_parent"])
else:
    print("Automatic batch has not started yet.")
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
    refresh_yolo_view()


if __name__ == "__main__":
    main()
