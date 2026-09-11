#!/usr/bin/env python3
"""17: Plot Spatial Maps"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

from scripts.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_marker_size(X: np.ndarray, Y: np.ndarray, fig_width: float, fig_height: float) -> float:
    """Compute optimal marker size for square markers to fill grid without gaps."""
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()

    x_sorted = np.sort(np.unique(X))
    y_sorted = np.sort(np.unique(Y))

    if len(x_sorted) > 1:
        x_spacing = np.median(np.diff(x_sorted))
    else:
        x_spacing = x_range / 100

    if len(y_sorted) > 1:
        y_spacing = np.median(np.diff(y_sorted))
    else:
        y_spacing = y_range / 100

    grid_spacing = min(x_spacing, y_spacing)

    dpi = 100

    data_per_inch_x = x_range / (fig_width * 0.8)
    data_per_inch_y = y_range / (fig_height * 0.8)

    data_per_inch = max(data_per_inch_x, data_per_inch_y)

    grid_in_inches = grid_spacing / data_per_inch
    grid_in_points = grid_in_inches * 72

    marker_size = (grid_in_points * 1.0) ** 2

    marker_size = max(0.5, min(marker_size, 50))

    return marker_size


def plot_practical_zone_map_for_mode(
    zone_mode: str,
    method: str,
    df: pd.DataFrame,
    config: dict,
    base_output_dir: Path,
    marker_size: float,
    deposit_mask: np.ndarray
):
    """Plot practical zone map for a specific zone mode."""
    n_folds = config['spatial_cv']['n_outer_folds']
    zone_dfs = []

    for fold_idx in range(n_folds):
        zone_path = (
            PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
            method / f'outer_fold_{fold_idx}' / 'practical_zone' / zone_mode / 'zone_assignments.csv'
        )
        if zone_path.exists():
            zone_df = pd.read_csv(zone_path)
            zone_dfs.append(zone_df)

    if not zone_dfs:
        logger.warning(f"No zone assignments found for {method}/{zone_mode}")
        return

    merged_zones = pd.concat(zone_dfs, ignore_index=True)

    df_copy = df.copy()
    if 'zone' in df_copy.columns:
        df_copy = df_copy.drop(columns=['zone'])

    df_with_zones = df_copy.merge(merged_zones[['index', 'zone']], on='index', how='left')

    if 'zone' not in df_with_zones.columns:
        logger.error(f"Zone column not found after merge for {zone_mode}")
        return

    if df_with_zones['zone'].isna().any():
        logger.warning(f"Some samples missing zone for {zone_mode}")
        df_with_zones['zone'] = df_with_zones['zone'].fillna(4)

    output_dir = base_output_dir / zone_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = config['visualization']['figure_formats']
    dpi = config['visualization']['dpi']
    zone_colors = config['visualization']['colormap_zone']
    n_zones = 5
    figsize = config['visualization'].get('figsize_map', (12, 10))

    X = df_with_zones['X'].values
    Y = df_with_zones['Y'].values
    deposit_size = max(5, marker_size * 0.3)

    mode_title = "Absolute Prob" if zone_mode == "prob" else "Lift Percentile"

    fig, ax = plt.subplots(figsize=figsize)

    zone_cmap = ListedColormap(zone_colors[:n_zones])
    scatter = ax.scatter(X, Y, c=df_with_zones['zone'], cmap=zone_cmap,
                         s=marker_size, marker='s', alpha=0.9,
                         edgecolors='none', vmin=0, vmax=n_zones-1, rasterized=True)

    if deposit_mask is not None and deposit_mask.sum() > 0:
        ax.scatter(X[deposit_mask], Y[deposit_mask],
                   c='black', s=deposit_size, marker='*',
                   edgecolors='white', linewidths=0.3, zorder=5, rasterized=True)

    legend_elements = []
    for z in range(n_zones):
        legend_elements.append(
            Patch(facecolor=zone_colors[z], edgecolor='none', label=f"Zone {z}")
        )

    if deposit_mask is not None and deposit_mask.sum() > 0:
        legend_elements.append(
            Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=12, markeredgecolor='white', markeredgewidth=0.5,
                   label=f'Deposits ({deposit_mask.sum()})')
        )

    legend = ax.legend(handles=legend_elements, loc='upper right',
                       fontsize=8, frameon=True, framealpha=1.0, ncol=1)
    legend.set_zorder(10)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    ax.grid(False)

    for fmt in formats:
        fig.savefig(output_dir / f"merged_practical_zone_map_{method}.{fmt}",
                    dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    logger.info(f"Saved practical zone map ({zone_mode}): {output_dir}")


def plot_spatial_maps(config: dict = None):
    """Plot spatial distribution maps with improved visualization."""
    logger.info("=" * 70)
    logger.info("Plotting Spatial Maps (Improved)")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    output_dir = PROJECT_ROOT / config['output_dirs']['figures'] / "spatial_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = config['methods']
    formats = config['visualization']['figure_formats']
    dpi = config['visualization']['dpi']
    zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])

    figsize = config['visualization'].get('figsize_map', (12, 10))

    for method in methods:
        logger.info(f"\nProcessing {method}...")

        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method
        merged_path = agg_dir / "merged_predictions.csv"

        if not merged_path.exists():
            logger.warning(f"Merged predictions not found for {method}")
            continue

        df = pd.read_csv(merged_path)
        logger.info(f"Loaded {len(df):,} samples")

        X = df['X'].values
        Y = df['Y'].values

        marker_size = compute_marker_size(X, Y, figsize[0], figsize[1])
        logger.info(f"Computed marker size: {marker_size:.2f}")

        deposit_mask = df['target'] == 1 if 'target' in df.columns else None
        deposit_size = max(5, marker_size * 0.3)

        prob_col = 'bootstrap_mean' if 'bootstrap_mean' in df.columns else 'calibrated_prob'
        logger.info(f"Using probability column: {prob_col}")

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(X, Y, c=df[prob_col],
                             cmap='RdYlGn_r',
                             s=marker_size, marker='s', alpha=0.9,
                             edgecolors='none', rasterized=True)

        if deposit_mask is not None and deposit_mask.sum() > 0:
            ax.scatter(X[deposit_mask], Y[deposit_mask],
                       c='black', s=deposit_size, marker='*',
                       edgecolors='white', linewidths=0.3, zorder=5, rasterized=True)
            legend_marker = Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor='black', markersize=12,
                                   markeredgecolor='white', markeredgewidth=0.5,
                                   label=f'Deposits ({deposit_mask.sum()})')
            legend = ax.legend(handles=[legend_marker], loc='upper right',
                               fontsize=9, frameon=True, framealpha=1.0)
            legend.set_zorder(10)

        cbar = fig.colorbar(scatter, ax=ax, label='Prospectivity (Bootstrap Mean)',
                            shrink=0.7, aspect=30)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')
        ax.grid(False)

        for fmt in formats:
            fig.savefig(output_dir / f"merged_prospectivity_map_{method}.{fmt}",
                        dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(X, Y, c=df['bootstrap_std'],
                             cmap='viridis',
                             s=marker_size, marker='s', alpha=0.9,
                             edgecolors='none', rasterized=True)

        if deposit_mask is not None and deposit_mask.sum() > 0:
            ax.scatter(X[deposit_mask], Y[deposit_mask],
                       c='red', s=deposit_size, marker='*',
                       edgecolors='white', linewidths=0.3, zorder=5, rasterized=True)
            legend_marker = Line2D([0], [0], marker='*', color='w',
                                   markerfacecolor='red', markersize=12,
                                   markeredgecolor='white', markeredgewidth=0.5,
                                   label=f'Deposits ({deposit_mask.sum()})')
            legend = ax.legend(handles=[legend_marker], loc='upper right',
                               fontsize=9, frameon=True, framealpha=1.0)
            legend.set_zorder(10)

        cbar = fig.colorbar(scatter, ax=ax, label='Bootstrap Std (Uncertainty)',
                            shrink=0.7, aspect=30)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')
        ax.grid(False)

        for fmt in formats:
            fig.savefig(output_dir / f"merged_uncertainty_map_{method}.{fmt}",
                        dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)

        in_set_1 = df['in_set_1'].astype(int).values

        conformal_colors = ['#9467bd', '#2ca02c']
        conformal_cmap = ListedColormap(conformal_colors)
        bounds = [-0.5, 0.5, 1.5]
        norm = BoundaryNorm(bounds, conformal_cmap.N)

        scatter = ax.scatter(X, Y, c=in_set_1, cmap=conformal_cmap,
                             norm=norm, s=marker_size, marker='s', alpha=0.9,
                             edgecolors='none', rasterized=True)

        if deposit_mask is not None and deposit_mask.sum() > 0:
            ax.scatter(X[deposit_mask], Y[deposit_mask],
                       c='black', s=deposit_size, marker='*',
                       edgecolors='white', linewidths=0.3, zorder=5, rasterized=True)

        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='none', label='Covered'),
            Patch(facecolor='#9467bd', edgecolor='none', label='Not Covered')
        ]
        if deposit_mask is not None and deposit_mask.sum() > 0:
            legend_elements.append(
                Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                       markersize=12, markeredgecolor='white', markeredgewidth=0.5,
                       label=f'Deposits ({deposit_mask.sum()})')
            )
        legend = ax.legend(handles=legend_elements, loc='upper right',
                           fontsize=9, frameon=True, framealpha=1.0)
        legend.set_zorder(10)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')
        ax.grid(False)

        for fmt in formats:
            fig.savefig(output_dir / f"merged_conformal_uq_map_{method}.{fmt}",
                        dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        for zone_mode in zone_modes:
            plot_practical_zone_map_for_mode(
                zone_mode=zone_mode,
                method=method,
                df=df,
                config=config,
                base_output_dir=output_dir,
                marker_size=marker_size,
                deposit_mask=deposit_mask
            )

    logger.info(f"\nFigures saved to: {output_dir}")


def main():
    plot_spatial_maps()


if __name__ == "__main__":
    main()
