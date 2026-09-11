#!/usr/bin/env python3
"""Generate journal-ready Figure 7."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from _figure_utils import (
    copy_pair,
    save_figure as _save_figure,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    aggregated_oof_dir: Path = project_root / "outputs" / "aggregated_oof"
    final_dir: Path = project_root / "figures"
    figure7_dir: Path = project_root / "outputs" / "figures" / "figure7"
    dpi: int = 600


N_ZONES: int = 5

METHODS: list[tuple[str, str, str]] = [
    ("xgboost", "XGBoost", "#1f77b4"),
    ("baggingpu_xgboost", "BaggingPU-XGBoost", "#ff7f0e"),
]


def load_cumulative_capture(method: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mean_cumulative, std_cumulative, n_zones) across 5 outer folds."""
    results_path = (
        Config.aggregated_oof_dir / method / "practical_zone" / "lift" / "all_fold_results.csv"
    )
    if not results_path.exists():
        raise FileNotFoundError(f"Missing lift-mode fold results: {results_path}")
    df = pd.read_csv(results_path)
    cumulative_cols = [
        f"zone_{z}_cumulative_pct"
        for z in range(N_ZONES)
        if f"zone_{z}_cumulative_pct" in df.columns
    ]
    if not cumulative_cols:
        raise ValueError(f"No cumulative_pct columns in {results_path}")
    return df[cumulative_cols].mean().to_numpy(), df[cumulative_cols].std().to_numpy(), len(cumulative_cols)


def create_figure7() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    max_zones = 0
    for method, label, color in METHODS:
        mean_cum, std_cum, n_zones = load_cumulative_capture(method)
        max_zones = max(max_zones, n_zones)
        x = np.arange(n_zones)
        ax.plot(
            x,
            mean_cum,
            marker="o",
            color=color,
            linewidth=2.2,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=label,
            zorder=4,
        )
        ax.fill_between(
            x,
            mean_cum - std_cum,
            mean_cum + std_cum,
            color=color,
            alpha=0.20,
            linewidth=0,
            label=f"{label} ±1 SD (5 folds)",
            zorder=3,
        )

    x_full = np.arange(max_zones)
    ax.axhline(y=91, color="#2ca02c", linestyle=":", linewidth=1.4, alpha=0.85,
               label="Cross-conformal coverage (≈91%)", zorder=2)

    ax.set_xlabel("Cumulative zone window", fontsize=10)
    ax.set_ylabel("Deposit capture (%)", fontsize=10)
    ax.set_title("Cumulative deposit capture by zone",
                 fontsize=11, fontweight="bold", pad=7)

    ax.set_xticks(x_full)
    ax.set_xticklabels([f"Zone 0–{z}" for z in range(max_zones)])
    ax.tick_params(axis="both", labelsize=8.5, direction="out", length=3, width=0.6)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(True, color="0.88", linewidth=0.55, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower right",
        fontsize=8.2,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.72",
        handlelength=2.0,
    )

    fig.tight_layout()

    provenance_stem = Config.figure7_dir / "deposit_capture_curve"
    _save_figure(fig, provenance_stem, dpi=Config.dpi, pad_inches=0.05)
    copy_pair(provenance_stem, Config.final_dir / "Figure_7")
    plt.close(fig)


def main() -> None:
    Config.final_dir.mkdir(parents=True, exist_ok=True)
    Config.figure7_dir.mkdir(parents=True, exist_ok=True)

    create_figure7()

    print("Figure 7 assets regenerated.")
    print(f"  Figure 7:  {Config.final_dir / 'Figure_7.pdf'}")
    for method, label, _ in METHODS:
        mean_cum, std_cum, n_z = load_cumulative_capture(method)
        cumulative_pct = ", ".join(
            f"Z0–{z}={mean_cum[z]:.1f}%±{std_cum[z]:.1f}%" for z in range(n_z)
        )
        print(f"  {label}: {cumulative_pct}")


if __name__ == "__main__":
    main()
