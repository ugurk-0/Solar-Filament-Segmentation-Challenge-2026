"""Export a self-contained Kaggle notebook from the current solution.py.

Run from any directory. The generated notebook calls the existing CLI; it does
not contain a second implementation of training or scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build(output_dir, root=ROOT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: (root / name).read_text(encoding="utf-8-sig")
               for name in ("solution.py", "requirements.txt")}
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
              for name, text in sources.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--",
                                *sources], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip())
    manifest = {"git_commit": commit, "source_files_modified": dirty,
                "sha256_utf8": hashes}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md("""# Solar Filament Segmentation Challenge 2026

Portable entry point generated from the repository's `solution.py`.
Attach the competition data, select a GPU for training, and configure the next
cell. All execution switches default to **False**. No checkpoints are bundled.
The research presentation notebook remains separate in the GitHub repository.
"""), code('''from pathlib import Path
import sys, json, hashlib, subprocess

PROJECT_DIR = Path("/kaggle/working/filament_source")
INPUT_ROOT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working/runs/fold0")
DATA_DIR = None  # Set explicitly if multiple annotation files are attached.
TEST_DIR = None  # Otherwise discovered beside the training directory.
FOLD = 0
EPOCHS = 60
LIMIT = 0  # Use 8 for a two-epoch smoke run; these checkpoints cannot submit.
TTA = True
INSTALL_DEPENDENCIES = False  # Enable with internet; restart kernel if needed.
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_TUNING = False
RUN_SUBMISSION = False
CHECKPOINTS = []  # Explicit paths to compatible checkpoints for submission.
'''), md("## Extract the exact source snapshot\n\nHashes identify the embedded source, including local edits at export time."),
             code("SOURCES = " + repr(sources) + "\nMANIFEST = " + repr(manifest) + '''
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
for name, text in SOURCES.items():
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == MANIFEST["sha256_utf8"][name]
    target = PROJECT_DIR / name
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Use a new PROJECT_DIR to preserve existing source: {target}")
    target.write_text(text, encoding="utf-8", newline="\\n")
print(json.dumps(MANIFEST, indent=2))
'''), md("## Environment\n\nInstall dependencies before running pipeline stages. Installation is optional because it can change Kaggle's preinstalled environment."),
             code('''if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(PROJECT_DIR / "requirements.txt")], check=True)
print("Interpreter:", sys.executable)
'''), md("## Locate inputs and run selected stages\n\nUse a new work directory for each training run. Evaluation prefers the best checkpoint; tuning uses the latest checkpoint. Submission uses explicit checkpoints and reads tuned settings from WORK_DIR."),
             code('''selected = RUN_TRAINING or RUN_EVALUATION or RUN_TUNING or RUN_SUBMISSION
training_data_needed = RUN_TRAINING or RUN_EVALUATION or RUN_TUNING
annotation_name = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
if training_data_needed:
    if DATA_DIR is None:
        matches = sorted(INPUT_ROOT.rglob(annotation_name))
        if len(matches) != 1:
            raise ValueError(f"Found {len(matches)} annotation files. Attach competition data and set DATA_DIR explicitly.")
        DATA_DIR = matches[0].parent
    DATA_DIR = Path(DATA_DIR)
    if not (DATA_DIR / annotation_name).is_file():
        raise FileNotFoundError(DATA_DIR / annotation_name)
if RUN_SUBMISSION:
    if TEST_DIR is None:
        matches = sorted(p for p in INPUT_ROOT.rglob("test_images") if p.is_dir())
        if len(matches) != 1:
            raise ValueError("Attach test data and set TEST_DIR explicitly.")
        TEST_DIR = matches[0]
    if not CHECKPOINTS or not all(Path(p).is_file() for p in CHECKPOINTS):
        raise FileNotFoundError("Set CHECKPOINTS to existing, compatible model files.")
    if LIMIT:
        raise ValueError("Disable LIMIT for submission; smoke checkpoints cannot submit.")

def run_stage(stage):
    args = [sys.executable, str(PROJECT_DIR / "solution.py"), stage,
            "--fold", str(FOLD), "--work-dir", str(WORK_DIR)]
    if stage != "submit":
        args += ["--data-dir", str(DATA_DIR), "--limit", str(LIMIT)]
    if not TTA:
        args += ["--no-tta"]
    if stage == "train":
        args += ["--epochs", str(EPOCHS)]
    if stage == "submit":
        args += ["--test-dir", str(TEST_DIR), "--ckpts", *map(str, CHECKPOINTS)]
    subprocess.run(args, cwd=PROJECT_DIR, check=True)

for enabled, stage in [(RUN_TRAINING, "train"), (RUN_EVALUATION, "eval"),
                       (RUN_TUNING, "tune"), (RUN_SUBMISSION, "submit")]:
    if enabled:
        run_stage(stage)
if not selected:
    print("Setup complete. Enable the stages you want in the configuration cell.")
'''), md("## Outputs\n\nSave the notebook version with its outputs to preserve runs. Download `submission.csv` when produced; this notebook does not automatically submit it for competition scoring."),
             code('''for name in [f"fold{FOLD}.pt", f"fold{FOLD}_best.pt",
             f"evaluation_fold{FOLD}.json", "best_postproc.json",
             "submission.csv", "submission_manifest.json"]:
    path = WORK_DIR / name
    if path.exists():
        print(path)
''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "source_manifest": manifest})
    nbf.validate(notebook)
    target = output_dir / "filament-kaggle.ipynb"
    nbf.write(notebook, target)
    (output_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist" / "kaggle")
    print(build(parser.parse_args().output_dir))
