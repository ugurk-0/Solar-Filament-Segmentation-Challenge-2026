# Repository organization

| Location | Role | Published to GitHub? |
|---|---|---|
| `solution.py`, `requirements.txt` | Main pipeline and environment | Yes |
| `notebook.ipynb`, `pq_experiment.py` | Research presentation and paired experiment | Yes |
| Root audit/report/training helpers | Existing experiment entry points; paths preserved | Yes, unless ignored |
| `main.tex` | Draft scientific report | Yes |
| `scripts/` | Packaging and repository utilities | Yes |
| `kaggle/` | Kaggle export/upload instructions | Yes |
| `docs/` | Usage and repository documentation | Yes |
| `tests/`, `.github/` | Regression checks and CI | Yes |
| `reports/` | Small written reports; review claims before publication | Yes, unless ignored |
| `MAGFiLO_1.0_Kaggle_2026/` | Local competition data | No |
| `runs/` | Checkpoints, evaluation artifacts, previews, submissions | No |
| `dist/kaggle/` | Generated portable notebook and source manifest | No; upload the notebook to Kaggle |

The existing Python entry points remain at the root to preserve notebook imports
and active experiment paths. New packaging utilities belong in `scripts/`.
Keep large outputs in `runs/` and rebuild upload bundles into `dist/`.

See [Kaggle upload](../kaggle/README.md) and [local auto-sync](AUTO_SYNC.md).
