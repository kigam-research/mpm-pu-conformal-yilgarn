#!/usr/bin/env python3
"""Generate journal-ready Figure 5 panels."""

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
    boundary_path: Path = project_root / "data" / "boundary" / "gadm41_AUS_0.shp"
    final_dir: Path = project_root / "figures"
    figure5_dir: Path = project_root / "outputs" / "figures" / "figure5"
    dpi: int = 600
    color_covered: str = "#41ab5d"
    color_not_covered: str = "#cbc9e2"


METHODS: list[tuple[str, str, str]] = [
    ("xgboost", "(a)", "XGBoost"),
    ("baggingpu_xgboost", "(b)", "BaggingPU-XGBoost"),
]


def load_predictions_oof(method: str) -> pd.DataFrame:
    csv_path = Config.aggregated_oof_dir / method / "merged_predictions.csv"
    df = pd.read_csv(csv_path)
    df["X_km"] = df["X"] / 1000.0
    df["Y_km"] = df["Y"] / 1000.0
    df["TMX"] = df["X"]
    df["TMY"] = df["Y"]
    return df


def coverage_grid(df: pd.DataFrame, column: str = "in_set_1") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = (
        pd.DataFrame({"TMX": df["TMX"], "TMY": df["TMY"], "value": df[column]})
        .pivot(index="TMY", columns="TMX", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    return table.columns.to_numpy() / 1000.0, table.index.to_numpy() / 1000.0, table.to_numpy()


def _legend_handles(deposit_count: int) -> list:
    return [
        Patch(facecolor=Config.color_covered, edgecolor="0.4", linewidth=0.3, label=r"Covered ($\mathbf{1}_{\hat{C}_\alpha}(x) = 1$)"),
        Patch(facecolor=Config.color_not_covered, edgecolor="0.4", linewidth=0.3, label=r"Not covered ($\mathbf{1}_{\hat{C}_\alpha}(x) = 0$)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
               markeredgecolor="white", markeredgewidth=0.5, markersize=10,
               label=f"Known Ni deposits (n={deposit_count:,})", linestyle="none"),
    ]


def _draw_coverage_panel(
    ax,
    df: pd.DataFrame,
    boundary_km,
    *,
    method_title: str,
    label: str,
    show_legend: bool,
) -> None:
    x, y, z = coverage_grid(df)
    extent = (x.min(), x.max(), y.min(), y.max())

    cmap = ListedColormap([Config.color_not_covered, Config.color_covered])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

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
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="0.7",
            handlelength=1.4,
        )


def create_individual_panel(method: str, label: str, method_title: str, boundary_km, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    _draw_coverage_panel(
        ax,
        df,
        boundary_km,
        method_title=method_title,
        label=label,
        show_legend=True,
    )
    fig.tight_layout()
    panel_stem = Config.figure5_dir / f"conformal_coverage_{method}"
    _save_figure(fig, panel_stem, dpi=Config.dpi, pad_inches=0.06)
    panel_letter = label.strip("()")
    copy_pair(panel_stem, Config.final_dir / f"Figure_5{panel_letter}")
    plt.close(fig)


def create_combined_figure5(boundary_km, dfs: dict[str, pd.DataFrame]) -> None:
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
        _draw_coverage_panel(
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
        ncol=3,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.7",
        handlelength=1.6,
        columnspacing=2.2,
        borderpad=0.6,
    )

    combined_stem = Config.figure5_dir / "Figure_5_combined"
    _save_figure(fig, combined_stem, dpi=Config.dpi, pad_inches=0.04)
    copy_pair(combined_stem, Config.final_dir / "Figure_5")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-ready Figure 5 assets.")
    parser.add_argument(
        "--skip-individual-panels",
        action="store_true",
        help="Skip per-method individual Figure_5a / Figure_5b export.",
    )
    args = parser.parse_args()

    Config.final_dir.mkdir(parents=True, exist_ok=True)
    Config.figure5_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for method, _label, _title in METHODS:
        dfs[method] = load_predictions_oof(method)

    boundary_km = _load_boundary_km(Config.boundary_path, dfs[METHODS[0][0]])

    if not args.skip_individual_panels:
        for method, label, method_title in METHODS:
            create_individual_panel(method, label, method_title, boundary_km, dfs[method])

    create_combined_figure5(boundary_km, dfs)

    print("Figure 5 assets regenerated.")
    print(f"  Figure 5(a): {Config.final_dir / 'Figure_5a.pdf'}")
    print(f"  Figure 5(b): {Config.final_dir / 'Figure_5b.pdf'}")
    print(f"  Combined:    {Config.final_dir / 'Figure_5.pdf'}")
    print(
        "  Coverage stats (OOF):  "
        + ", ".join(
            f"{method} {(dfs[method]['in_set_1'] == 1).mean() * 100:.1f}%"
            for method, _, _ in METHODS
        )
    )


if __name__ == "__main__":
    main()
