"""Run the second PQ-guided fine-tuning cycle without overwriting prior artifacts."""
from pathlib import Path

import pq_experiment
import solution


ROOT = Path("runs/pq_research_20260926")
cfg = pq_experiment.config(ROOT / "training_round2", fold=1)
cfg.epochs = 3
cfg.lr = 2e-5
cfg.ema_decay = 0.99
cfg.eval_every = 1
cfg.eval_max_images = 16
cfg.init_ckpt = str(ROOT / "training" / "fold1_best.pt")
cfg.instance_method = "components"
cfg.thresh = 0.65
cfg.min_area = 200

solution.train(cfg)
