#!/usr/bin/env python3
"""Generate journal-ready Figure 6 panels."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _figure_utils import (
    add_north_arrow,
    add_panel_label,
    add_scale_bar,
    copy_pair,
    format_map_axis,
    load_boundary_km as _load_boundary_km,
    plot_boundary,
    save_figure as _save_figure,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    aggregated_oof_dir: Path = project_root / "outputs" / "aggregated_oof"
    test_eval_oof_dir: Path = project_root / "outputs" / "test_evaluation_oof"
    boundary_path: Path = project_root / "data" / "boundary" / "gadm41_AUS_0.shp"
    final_dir: Path = project_root / "figures"
    figure6_dir: Path = project_root / "outputs" / "figures" / "figure6"
    dpi: int = 600


ZONE_COLORS: tuple[str, ...] = ("#d62728", "#ff7f0e", "#ffbb78", "#2ca02c", "#9467bd")
ZONE_LABELS: tuple[str, ...] = ("Zone 0", "Zone 1", "Zone 2", "Zone 3", "Zone 4 (excluded)")
N_ZONES: int = 5

METHODS: list[tuple[str, str, str]] = [
    ("xgboost", "(a)", "XGBoost"),
    ("baggingpu_xgboost", "(b)", "BaggingPU-XGBoost"),
]


def load_lift_zone_assignments(method: str) -> pd.DataFrame:
    """Load OOF lift-mode per-cell zone assignments by concatenating outer folds."""
    fold_csvs = sorted(
        (Config.test_eval_oof_dir / method).glob("outer_fold_*/practical_zone/lift/zone_assignments.csv")
    )
    if not fold_csvs:
        raise FileNotFoundError(f"No outer-fold lift zone_assignments.csv found for {method}")
    frames = [pd.read_csv(p) for p in fold_csvs]
    merged = pd.concat(frames, ignore_index=True)
    if not {"index", "X", "Y", "zone"}.issubset(merged.columns):
        raise ValueError(f"Unexpected columns in zone_assignments: {merged.columns.tolist()}")
    return merged


def load_method_data(method: str) -> pd.DataFrame:
    """Load OOF predictions + lift-mode per-cell zones merged on `index`."""
    pred_path = Config.aggregated_oof_dir / method / "merged_predictions.csv"
    preds = pd.read_csv(pred_path)
    zones = load_lift_zone_assignments(method)
    if "zone" in preds.columns:
        preds = preds.drop(columns=["zone"])
    df = preds.merge(zones[["index", "zone"]], on="index", how="left")
    if df["zone"].isna().any():
        df["zone"] = df["zone"].fillna(N_ZONES - 1).astype(int)
    df["X_km"] = df["X"] / 1000.0
    df["Y_km"] = df["Y"] / 1000.0
    df["TMX"] = df["X"]
    df["TMY"] = df["Y"]
    return df


def zone_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = (
        pd.DataFrame({"TMX": df["TMX"], "TMY": df["TMY"], "value": df["zone"]})
        .pivot(index="TMY", columns="TMX", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    return table.columns.to_numpy() / 1000.0, table.index.to_numpy() / 1000.0, table.to_numpy()


def _legend_handles(deposit_count: int) -> list:
    handles = [
        Patch(facecolor=ZONE_COLORS[i], edgecolor="0.4", linewidth=0.3, label=ZONE_LABELS[i])
        for i in range(N_ZONES)
    ]
    handles.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markeredgecolor="white", markeredgewidth=0.5, markersize=10,
               label=f"Known Ni deposits (n={deposit_count:,})", linestyle="none")
    )
    return handles


def _draw_zone_panel(
    ax,
    df: pd.DataFrame,
    boundary_km,
    *,
    method_title: str,
    label: str,
    show_legend: bool,
) -> None:
    x, y, z = zone_grid(df)
    extent = (x.min(), x.max(), y.min(), y.max())

    cmap = ListedColormap(list(ZONE_COLORS))
    boundaries = [i - 0.5 for i in range(N_ZONES + 1)]
    norm = BoundaryNorm(boundaries, cmap.N)

    ax.imshow(
        np.ma.masked_invalid(z),
        extent=extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )

    plot_boundary(ax, boundary_km, facecolor=None, edgecolor="0.15", linewidth=0.6, zorder=5)
    format_map_axis(ax, df)

    deposits = df.loc[df["target"] == 1]
    ax.scatter(
        deposits["X_km"],
        deposits["Y_km"],
        s=14,
        c="black",
        marker="*",
        edgecolors="white",
        linewidths=0.4,
        alpha=0.95,
        rasterized=True,
        zorder=8,
    )

    add_north_arrow(ax)
    add_scale_bar(ax, length_km=200.0, location=(0.08, 0.075))
    add_panel_label(ax, label)
    ax.set_title(method_title, fontsize=11, fontweight="bold", pad=7)

    if show_legend:
        handles = _legend_handles(len(deposits))
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7.5,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="0.7",
            handlelength=1.2,
        )


def create_individual_panel(method: str, label: str, method_title: str, boundary_km, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    _draw_zone_panel(
        ax,
        df,
        boundary_km,
        method_title=method_title,
        label=label,
        show_legend=True,
    )
    fig.tight_layout()
    panel_stem = Config.figure6_dir / f"practical_zone_lift_{method}"
    _save_figure(fig, panel_stem, dpi=Config.dpi, pad_inches=0.06)
    panel_letter = label.strip("()")
    copy_pair(panel_stem, Config.final_dir / f"Figure_6{panel_letter}")
    plt.close(fig)


def create_combined_figure6(boundary_km, dfs: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(13.4, 6.6))
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.045,
        left=0.035,
        right=0.985,
        bottom=0.18,
        top=0.945,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, (method, label, method_title) in zip(axes, METHODS):
        _draw_zone_panel(
            ax,
            dfs[method],
            boundary_km,
            method_title=method_title,
            label=label,
            show_legend=False,
        )

    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()
    legend_center_x = (pos_a.x0 + pos_b.x1) / 2.0

    deposit_count = int((dfs[METHODS[0][0]]["target"] == 1).sum())
    handles = _legend_handles(deposit_count)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(legend_center_x, 0.03),
        ncol=6,
        fontsize=9.5,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.7",
        handlelength=1.4,
        columnspacing=1.6,
        borderpad=0.55,
    )

    combined_stem = Config.figure6_dir / "Figure_6_combined"
    _save_figure(fig, combined_stem, dpi=Config.dpi, pad_inches=0.04)
    copy_pair(combined_stem, Config.final_dir / "Figure_6")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-ready Figure 6 assets.")
    parser.add_argument(
        "--skip-individual-panels",
        action="store_true",
        help="Skip per-method individual Figure_6a / Figure_6b export.",
    )
    args = parser.parse_args()

    Config.final_dir.mkdir(parents=True, exist_ok=True)
    Config.figure6_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for method, _label, _title in METHODS:
        dfs[method] = load_method_data(method)

    boundary_km = _load_boundary_km(Config.boundary_path, dfs[METHODS[0][0]])

    if not args.skip_individual_panels:
        for method, label, method_title in METHODS:
            create_individual_panel(method, label, method_title, boundary_km, dfs[method])

    create_combined_figure6(boundary_km, dfs)

    print("Figure 6 assets regenerated.")
    print(f"  Figure 6(a): {Config.final_dir / 'Figure_6a.pdf'}")
    print(f"  Figure 6(b): {Config.final_dir / 'Figure_6b.pdf'}")
    print(f"  Combined:    {Config.final_dir / 'Figure_6.pdf'}")
    for method, _, _ in METHODS:
        dist = dfs[method]["zone"].value_counts(normalize=True).sort_index() * 100
        msg = ", ".join(f"Z{int(z)}={dist.get(z, 0):.2f}%" for z in range(N_ZONES))
        print(f"  Lift zones ({method}): {msg}")


if __name__ == "__main__":
    main()
