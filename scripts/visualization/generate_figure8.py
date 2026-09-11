#!/usr/bin/env python3
"""Generate journal-ready Figure 8 panels."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _figure_utils import (
    add_panel_label,
    copy_pair,
    save_figure as _save_figure,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    aggregated_oof_dir: Path = project_root / "outputs" / "aggregated_oof"
    test_eval_oof_dir: Path = project_root / "outputs" / "test_evaluation_oof"
    final_dir: Path = project_root / "figures"
    figure8_dir: Path = project_root / "outputs" / "figures" / "figure8"
    dpi: int = 600
    block_size_m: int = 50_000


ZONE_COLORS: tuple[str, ...] = ("#d62728", "#ff7f0e", "#ffbb78", "#2ca02c", "#9467bd")
ZONE_LABELS: tuple[str, ...] = (
    "Zone 0: Immediate",
    "Zone 1: Priority",
    "Zone 2: Follow-up",
    "Zone 3: Potential",
    "Zone 4: Excluded",
)
N_ZONES: int = 5

PANELS: list[tuple[str, int, int, str, str]] = [
    ("xgboost",            2, 2, "(a)", "XGBoost"),
    ("xgboost",            4, 1, "(b)", "XGBoost"),
    ("baggingpu_xgboost",  2, 2, "(c)", "BaggingPU-XGBoost"),
    ("baggingpu_xgboost",  4, 1, "(d)", "BaggingPU-XGBoost"),
]


def load_method_fold_zones(method: str, fold: int) -> pd.DataFrame:
    """Load per-cell lift zones for a (method, fold) plus the target indicator."""
    zone_path = (
        Config.test_eval_oof_dir / method / f"outer_fold_{fold}" / "practical_zone" / "lift" / "zone_assignments.csv"
    )
    if not zone_path.exists():
        raise FileNotFoundError(f"Missing zone assignments: {zone_path}")
    zones = pd.read_csv(zone_path)
    preds = pd.read_csv(Config.aggregated_oof_dir / method / "merged_predictions.csv")
    df = zones.merge(preds[["index", "target"]], on="index", how="left")
    x0, y0 = float(preds["X"].min()), float(preds["Y"].min())
    df["block_x"] = ((df["X"] - x0) // Config.block_size_m).astype(int)
    df["block_y"] = ((df["Y"] - y0) // Config.block_size_m).astype(int)
    df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)
    df["X_km"] = df["X"] / 1000.0
    df["Y_km"] = df["Y"] / 1000.0
    return df


def select_top_deposit_block(df: pd.DataFrame, rank: int) -> str:
    """Return the block_id of the block ranked `rank` by deposit count (rank=1 → top)."""
    deposits_per_block = (
        df.loc[df["target"] == 1].groupby("block_id").size().sort_values(ascending=False)
    )
    if len(deposits_per_block) < rank:
        raise ValueError(f"Cannot pick rank-{rank} block: only {len(deposits_per_block)} blocks contain deposits.")
    return deposits_per_block.index[rank - 1]


def subset_to_block(df: pd.DataFrame, block_id: str) -> pd.DataFrame:
    return df.loc[df["block_id"] == block_id].copy()


def zone_grid_block(df_block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = (
        df_block.pivot(index="Y", columns="X", values="zone")
        .sort_index()
        .sort_index(axis=1)
    )
    return table.columns.to_numpy() / 1000.0, table.index.to_numpy() / 1000.0, table.to_numpy()


def _legend_handles_no_count() -> list:
    handles = [
        Patch(facecolor=ZONE_COLORS[i], edgecolor="0.4", linewidth=0.3, label=ZONE_LABELS[i])
        for i in range(N_ZONES)
    ]
    handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markeredgecolor="white", markeredgewidth=0.5, markersize=10,
               label="Positive analysis cells", linestyle="none")
    )
    return handles


def _draw_block_panel(
    ax,
    df_block: pd.DataFrame,
    *,
    label: str,
    method_title: str,
    fold_num: int,
) -> None:
    x, y, z = zone_grid_block(df_block)
    half_km = 5.0 / 2.0
    x_edges = np.concatenate([x - half_km, [x[-1] + half_km]])
    y_edges = np.concatenate([y - half_km, [y[-1] + half_km]])
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])

    cmap = ListedColormap(list(ZONE_COLORS))
    norm = BoundaryNorm([i - 0.5 for i in range(N_ZONES + 1)], cmap.N)

    ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(z),
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.55,
        shading="flat",
        rasterized=True,
    )

    deposits = df_block.loc[df_block["target"] == 1]
    ax.scatter(
        deposits["X_km"],
        deposits["Y_km"],
        s=140,
        c="black",
        marker="*",
        edgecolors="white",
        linewidths=0.9,
        alpha=0.96,
        rasterized=True,
        zorder=8,
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.tick_params(axis="both", labelsize=8, direction="out", length=3, width=0.6)
    ax.set_xlabel("Easting (km)", fontsize=9)
    ax.set_ylabel("Northing (km)", fontsize=9)
    ax.grid(False)

    add_panel_label(ax, label)
    ax.set_title(f"{method_title} — Fold {fold_num} (n={len(deposits)} positive cells)", fontsize=10.5, fontweight="bold", pad=6)


def create_individual_panel(method: str, fold: int, block_rank: int, label: str, method_title: str) -> None:
    df = load_method_fold_zones(method, fold)
    block_id = select_top_deposit_block(df, block_rank)
    df_block = subset_to_block(df, block_id)

    fig, ax = plt.subplots(figsize=(6.3, 6.4))
    _draw_block_panel(ax, df_block, label=label, method_title=method_title, fold_num=fold)
    handles = _legend_handles_no_count()
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="0.7",
        handlelength=1.2,
    )
    fig.tight_layout()
    panel_stem = Config.figure8_dir / f"block_{method}_fold{fold}"
    _save_figure(fig, panel_stem, dpi=Config.dpi, pad_inches=0.06)
    panel_letter = label.strip("()")
    copy_pair(panel_stem, Config.final_dir / f"Figure_8{panel_letter}")
    plt.close(fig)


def create_combined_figure8() -> None:
    fig = plt.figure(figsize=(12.6, 12.8))
    gs = fig.add_gridspec(
        2, 2,
        wspace=0.20,
        hspace=0.22,
        left=0.06,
        right=0.985,
        bottom=0.10,
        top=0.965,
    )
    axes_flat = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, (method, fold, block_rank, label, method_title) in zip(axes_flat, PANELS):
        df = load_method_fold_zones(method, fold)
        block_id = select_top_deposit_block(df, block_rank)
        df_block = subset_to_block(df, block_id)
        _draw_block_panel(ax, df_block, label=label, method_title=method_title, fold_num=fold)

    fig.canvas.draw()
    pos_a = axes_flat[0].get_position()
    pos_b = axes_flat[1].get_position()
    legend_center_x = (pos_a.x0 + pos_b.x1) / 2.0

    handles = _legend_handles_no_count()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(legend_center_x, 0.02),
        ncol=6,
        fontsize=9.5,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.7",
        handlelength=1.4,
        columnspacing=1.5,
        borderpad=0.55,
    )

    combined_stem = Config.figure8_dir / "Figure_8_combined"
    _save_figure(fig, combined_stem, dpi=Config.dpi, pad_inches=0.04)
    copy_pair(combined_stem, Config.final_dir / "Figure_8")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-ready Figure 8 assets.")
    parser.add_argument(
        "--skip-individual-panels",
        action="store_true",
        help="Skip per-panel individual Figure_8{a,b,c,d} export.",
    )
    args = parser.parse_args()

    Config.final_dir.mkdir(parents=True, exist_ok=True)
    Config.figure8_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_individual_panels:
        for method, fold, block_rank, label, method_title in PANELS:
            create_individual_panel(method, fold, block_rank, label, method_title)

    create_combined_figure8()

    print("Figure 8 assets regenerated.")
    print(f"  Figure 8(a): {Config.final_dir / 'Figure_8a.pdf'} — XGB Fold 2 (top-2 deposit block)")
    print(f"  Figure 8(b): {Config.final_dir / 'Figure_8b.pdf'} — XGB Fold 4 (top-1 deposit block)")
    print(f"  Figure 8(c): {Config.final_dir / 'Figure_8c.pdf'} — BAG Fold 2 (top-2 deposit block)")
    print(f"  Figure 8(d): {Config.final_dir / 'Figure_8d.pdf'} — BAG Fold 4 (top-1 deposit block)")
    print(f"  Combined:    {Config.final_dir / 'Figure_8.pdf'}")


if __name__ == "__main__":
    main()
