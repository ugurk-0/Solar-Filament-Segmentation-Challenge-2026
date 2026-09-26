# Kaggle upload

Build a portable notebook from the current source:

```bash
python -m pip install -r requirements.txt
python scripts/build_kaggle.py
```

Upload **`dist/kaggle/filament-kaggle.ipynb`** through Kaggle's notebook import.
The generated notebook embeds `solution.py` and `requirements.txt`; no Git clone
or separate source dataset is needed inside Kaggle. Rebuild after source edits.
`source-manifest.json` records the source hashes, Git commit, and whether the
embedded files contain uncommitted changes.

1. Attach the competition data to the notebook and accept the competition rules.
2. Choose a GPU for training. Select the intended Python environment.
3. Run the configuration and extraction cells. Enable dependency installation
   if needed; it requires internet and may require a kernel restart.
4. Choose a new `WORK_DIR`, a fold, and the stages to run. A smoke run uses
   `LIMIT = 8`, `EPOCHS = 2`, and `TTA = False`.
5. For final training, set `LIMIT = 0`. Preserve the resulting checkpoint outputs.
6. For inference, attach checkpoints as an input or use checkpoints from the
   current session, set `CHECKPOINTS`, and enable `RUN_SUBMISSION`. Supply
   `best_postproc.json` in `WORK_DIR` to apply previously tuned settings.
7. Save a notebook version with outputs. Download the generated CSV for submission.

All expensive stages default to disabled. Input discovery requires exactly one
matching training annotation file or test image directory; set `DATA_DIR` and
`TEST_DIR` explicitly if multiple inputs match. For inference alone, training
annotations are not required.

The root `notebook.ipynb` is the research presentation and depends on local
experiment artifacts. The generated Kaggle notebook is the portable CLI entry
point. Neither contains datasets, model weights, or measured results.

See [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks)
for data attachments, compute settings, and saved outputs. A notebook upload
does not automatically submit predictions to the competition.
