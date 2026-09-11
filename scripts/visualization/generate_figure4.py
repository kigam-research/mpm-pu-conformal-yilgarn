#!/usr/bin/env python3
"""Generate journal-ready Figure 4 panels."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    aggregated_dir: Path = project_root / "outputs" / "aggregated"
    boundary_path: Path = project_root / "data" / "boundary" / "gadm41_AUS_0.shp"
    final_dir: Path = project_root / "figures"
    figure4_dir: Path = project_root / "outputs" / "figures" / "figure4"
    dpi: int = 600
    cmap: str = "RdYlGn_r"
    vmin: float = 0.0
    vmax: float = 1.0
    cbar_label: str = "Prospectivity (bootstrap mean probability)"


METHODS: list[tuple[str, str, str]] = [
    ("xgboost", "(a)", "XGBoost"),
    ("baggingpu_xgboost", "(b)", "BaggingPU-XGBoost"),
]


def load_predictions(method: str) -> pd.DataFrame:
    csv_path = Config.aggregated_dir / method / "merged_predictions.csv"
    df = pd.read_csv(csv_path)
    df["X_km"] = df["X"] / 1000.0
    df["Y_km"] = df["Y"] / 1000.0
    df["TMX"] = df["X"]
    df["TMY"] = df["Y"]
    return df


def prospectivity_grid(df: pd.DataFrame, column: str = "bootstrap_mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = (
        pd.DataFrame({"TMX": df["TMX"], "TMY": df["TMY"], "value": df[column]})
        .pivot(index="TMY", columns="TMX", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    return table.columns.to_numpy() / 1000.0, table.index.to_numpy() / 1000.0, table.to_numpy()


def _draw_prospectivity_panel(
    ax,
    df: pd.DataFrame,
    boundary_km,
    *,
    method_title: str,
    label: str,
    show_legend: bool,
) -> "plt.cm.ScalarMappable":
    x, y, z = prospectivity_grid(df)
    extent = (x.min(), x.max(), y.min(), y.max())

    image = ax.imshow(
        np.ma.masked_invalid(z),
        extent=extent,
        origin="lower",
        cmap=Config.cmap,
        vmin=Config.vmin,
        vmax=Config.vmax,
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
        label=f"Known Ni deposits (n={len(deposits):,})",
        zorder=8,
    )

    add_north_arrow(ax)
    add_scale_bar(ax, length_km=200.0, location=(0.08, 0.075))
    add_panel_label(ax, label)
    ax.set_title(method_title, fontsize=11, fontweight="bold", pad=7)

    if show_legend:
        ax.legend(
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="0.7",
            markerscale=1.6,
        )

    return image


def create_individual_panel(method: str, label: str, method_title: str, boundary_km, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    image = _draw_prospectivity_panel(
        ax,
        df,
        boundary_km,
        method_title=method_title,
        label=label,
        show_legend=True,
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.8, aspect=28, pad=0.02)
    cbar.set_label(Config.cbar_label, fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=7.8, length=2.4, width=0.5)

    fig.tight_layout()
    panel_stem = Config.figure4_dir / f"prospectivity_{method}"
    _save_figure(fig, panel_stem, dpi=Config.dpi, pad_inches=0.06)
    panel_letter = label.strip("()")
    copy_pair(panel_stem, Config.final_dir / f"Figure_4{panel_letter}")
    plt.close(fig)


def create_combined_figure4(boundary_km, dfs: dict[str, pd.DataFrame]) -> None:
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

    last_image = None
    for ax, (method, label, method_title) in zip(axes, METHODS):
        last_image = _draw_prospectivity_panel(
            ax,
            dfs[method],
            boundary_km,
            method_title=method_title,
            label=label,
            show_legend=(method == METHODS[0][0]),
        )

    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()
    cbar_left = pos_a.x0
    cbar_right = pos_b.x1
    cbar_width = cbar_right - cbar_left
    cbar_height = 0.026
    cbar_bottom = 0.085

    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(last_image, cax=cax, orientation="horizontal")
    cbar.set_label(Config.cbar_label, fontsize=9.5, labelpad=4)
    cbar.ax.tick_params(labelsize=8, length=2.6, width=0.5)

    combined_stem = Config.figure4_dir / "Figure_4_combined"
    _save_figure(fig, combined_stem, dpi=Config.dpi, pad_inches=0.04)
    copy_pair(combined_stem, Config.final_dir / "Figure_4")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-ready Figure 4 assets.")
    parser.add_argument(
        "--skip-individual-panels",
        action="store_true",
        help="Skip per-method individual Figure_4a / Figure_4b export (combined Figure_4 still produced).",
    )
    args = parser.parse_args()

    Config.final_dir.mkdir(parents=True, exist_ok=True)
    Config.figure4_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for method, _label, _title in METHODS:
        dfs[method] = load_predictions(method)

    boundary_km = _load_boundary_km(Config.boundary_path, dfs[METHODS[0][0]])

    if not args.skip_individual_panels:
        for method, label, method_title in METHODS:
            create_individual_panel(method, label, method_title, boundary_km, dfs[method])

    create_combined_figure4(boundary_km, dfs)

    print("Figure 4 assets regenerated.")
    print(f"  Figure 4(a): {Config.final_dir / 'Figure_4a.pdf'}")
    print(f"  Figure 4(b): {Config.final_dir / 'Figure_4b.pdf'}")
    print(f"  Combined:    {Config.final_dir / 'Figure_4.pdf'}")


if __name__ == "__main__":
    main()
