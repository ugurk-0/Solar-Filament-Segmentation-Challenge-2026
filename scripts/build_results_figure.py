"""Render the public results figure from the small, versioned metrics snapshot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "reports/results_summary.json").read_text())
    rows = data["assessment"]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    bars = ax.barh([r["label"] for r in rows], [r["mean_pq"] for r in rows],
                   color=["#94a3b8", "#64748b", "#38bdf8", "#0284c7"])
    ax.bar_label(bars, fmt="%.4f", padding=5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.32)
    ax.set_xlabel("Mean per-observation Panoptic Quality (higher is better)")
    ax.set_title("Solar filament segmentation · 85-observation local assessment", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.04, "One fold and seed; no TTA. Reused assessment split; not a leaderboard score.", fontsize=9)
    output = ROOT / "docs/assets/results.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
