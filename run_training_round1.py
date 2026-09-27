"""Reproduce the first four-epoch warm-start training cycle, for sake of logical consistency of the repo."""

from pathlib import Path

import pq_experiment
import solution


ROOT = Path("runs/pq_research_20260926")
cfg = pq_experiment.config(ROOT / "training", fold=1)
cfg.epochs = 4
cfg.lr = 1e-4
cfg.ema_decay = 0.9
cfg.eval_every = 1
cfg.eval_max_images = 16
cfg.init_ckpt = "runs/corrected_loss_fold1/fold1.pt"

solution.train(cfg)
