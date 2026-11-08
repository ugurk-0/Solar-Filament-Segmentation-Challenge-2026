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
Earlier methods remain documented in the experiment reports.'''),
        ('code', '''from pathlib import Path
assert Path("yolo_dashboard.py").is_file(), "Start the notebook kernel in the repository root"
from yolo_dashboard import display_snapshot, dashboard
_ = display_snapshot(Path("runs/yolo11n_20261003"))'''),
        ('code', '''YOLO_PANEL = dashboard(Path("runs"), auto_refresh=False, interval=15)
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
    """Build the final presentation; keep experiment evidence in reports/runs."""
    backup = Path("runs/final_cleanup_20261108/notebook_before.ipynb")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copyfile("notebook.ipynb", backup)
    cells = [nbf.v4.new_markdown_cell("""# Solar filament segmentation: final YOLO submission

The selected model is **YOLO11n-seg, epoch 17**, with confidence 0.50 and minimum
area 128, without TTA. Local mean PQ is 0.3199 and pooled PQ 0.3319 on the reused
85-observation assessment. Kaggle submission **58139775** was accepted on
8 November 2026 and scored **0.28**, up from the prior **0.23**. The status cell reads the verified Kaggle result.

This notebook displays the selected model's training history, predictions and
submission evidence. It does not launch training or upload files. Rejected
implementations were retired; reports and run artifacts preserve their results.
See `reports/final_submission_20261108.md` for the cleanup ledger.
""")]
    cells += yolo_cells()
    cells.append(nbf.v4.new_markdown_cell("## Submission evidence"))
    cells.append(nbf.v4.new_code_cell('''
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, FileLink
assert Path("submit_yolo.py").is_file(), "Run from the repository root"
status_path = Path("reports/final_submission_status.json")
if status_path.exists():
    display(pd.Series(json.loads(status_path.read_text())))
submission = Path("submissions/final_20261108")
manifest = submission / "submission_manifest.json"
if manifest.exists():
    result = json.loads(manifest.read_text())
    display({k: result[k] for k in ("checkpoint_sha256", "inference", "n_images", "n_instances", "csv_sha256", "validation")})
for name in ("submission.csv", "submission_description.txt"):
    path = submission / name
    if path.exists(): display(FileLink(str(path)))
'''.strip()))
    cells.append(nbf.v4.new_markdown_cell("""## Reproduction and limitations

Install `requirements-yolo.txt` and run `python submit_yolo.py --output submissions/reproduced`
from the root with the retained checkpoint/protocol and competition images available.
The manifest verifies the evaluated inference behavior, checkpoint identity and every
RLE mask. A new output directory preserves the accepted submission.

The assessment was reused during research. Local PQ is not a leaderboard score.
The previous submission's API-verified public score is 0.23. Historical experiment
results remain in `reports/`; the retained regression suite runs with
`python -m pytest tests -q`. No test labels were used for selection.
"""))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}})
    nbf.validate(notebook)
    nbf.write(notebook, "notebook.ipynb")


if __name__ == "__main__":
    main()
