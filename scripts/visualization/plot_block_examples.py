#!/usr/bin/env python3
"""18: Plot Block Examples"""

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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scripts.utils import load_config, load_outer_folds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_representative_blocks(df: pd.DataFrame, fold_assignments: pd.DataFrame, top_n: int = 2) -> list:
    """Find the original spatial block_ids with the most deposits."""
    df = df.copy()
    df = df.merge(fold_assignments[['index', 'block_id']], on='index', how='left')

    block_deposits = df[df['target'] == 1].groupby('block_id').size()

    if len(block_deposits) == 0:
        block_samples = df.groupby('block_id').size()
        top_blocks = block_samples.nlargest(top_n).index.tolist()
    else:
        top_blocks = block_deposits.nlargest(top_n).index.tolist()

    return top_blocks


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


    data_per_inch_x = x_range / (fig_width * 0.8)
    data_per_inch_y = y_range / (fig_height * 0.8)

    data_per_inch = max(data_per_inch_x, data_per_inch_y)

    grid_in_inches = grid_spacing / data_per_inch
    grid_in_points = grid_in_inches * 72

    marker_size = (grid_in_points * 1.0) ** 2

    marker_size = max(0.5, marker_size)

    return marker_size


def plot_block_examples(config: dict = None, method: str = None):
    """Plot representative block examples - Practical Zone Map for Prob & Lift modes."""
    logger.info("=" * 70)
    logger.info("Plotting Block Examples (Practical Zone Map, 5 Zones rel_IQR 30%)")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    if method is None:
        method = config['methods'][0]

    logger.info(f"Method: {method}")

    base_output_dir = PROJECT_ROOT / config['output_dirs']['figures'] / "block_examples" / method

    n_folds = config['spatial_cv']['n_outer_folds']
    formats = config['visualization']['figure_formats']
    dpi = config['visualization']['dpi']
    zone_colors = config['visualization']['colormap_zone']
    zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])
    n_zones = 5

    zone_names = {
        0: "Immediate",
        1: "Priority",
        2: "Follow-up",
        3: "Potential",
        4: "Excluded"
    }

    fold_assignments = load_outer_folds(config)

    for fold_idx in range(n_folds):
        logger.info(f"\nProcessing fold {fold_idx}...")

        eval_dir = (
            PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
            method / f'outer_fold_{fold_idx}'
        )
        predictions_path = eval_dir / "test_predictions.csv"

        if not predictions_path.exists():
            logger.warning(f"Predictions not found for fold {fold_idx}")
            continue

        df = pd.read_csv(predictions_path)

        top_blocks = find_representative_blocks(df, fold_assignments, top_n=2)

        df_with_block = df.merge(
            fold_assignments[['index', 'block_id']], on='index', how='left'
        )

        for zone_mode in zone_modes:
            logger.info(f"  Zone mode: {zone_mode.upper()}")

            zone_path = (
                eval_dir / 'practical_zone' / zone_mode / 'zone_assignments.csv'
            )

            if not zone_path.exists():
                logger.warning(f"Zone assignments not found: {zone_path}")
                continue

            zone_df = pd.read_csv(zone_path)

            df_for_merge = df_with_block.copy()
            if 'zone' in df_for_merge.columns:
                df_for_merge = df_for_merge.drop(columns=['zone'])

            df_with_zones = df_for_merge.merge(
                zone_df[['index', 'zone']], on='index', how='left'
            )

            if 'zone' not in df_with_zones.columns:
                logger.warning(f"Zone column not found after merge for {zone_mode}")
                continue

            if df_with_zones['zone'].isna().any():
                df_with_zones['zone'] = df_with_zones['zone'].fillna(4)

            mode_title = "Absolute Prob" if zone_mode == "prob" else "Lift Percentile"

            for rank, block_id in enumerate(top_blocks, start=1):
                rank_name = f"top{rank}_deposit"
                output_dir = base_output_dir / zone_mode / rank_name
                output_dir.mkdir(parents=True, exist_ok=True)

                block_df = df_with_zones[df_with_zones['block_id'] == block_id].copy()

                if len(block_df) < 10:
                    logger.warning(f"Too few samples in {rank_name} block for fold {fold_idx}")
                    continue

                n_deposits = int(block_df['target'].sum())
                logger.info(f"    {rank_name}: {len(block_df)} samples, {n_deposits} deposits")

                figsize = (12, 10)
                fig, ax = plt.subplots(figsize=figsize)

                X = block_df['X'].values
                Y = block_df['Y'].values
                deposit_mask = block_df['target'] == 1

                marker_size = compute_marker_size(X, Y, figsize[0], figsize[1])
                deposit_size = max(10, marker_size * 0.3)

                zone_cmap = ListedColormap(zone_colors[:n_zones])
                scatter = ax.scatter(X, Y, c=block_df['zone'], cmap=zone_cmap,
                                     s=marker_size, marker='s', alpha=0.9,
                                     edgecolors='none', vmin=0, vmax=n_zones-1, rasterized=True)

                ax.scatter(X[deposit_mask], Y[deposit_mask],
                           marker='*', s=deposit_size, c='black',
                           edgecolors='white', linewidths=0.5, zorder=5, rasterized=True)

                legend_elements = []
                for z in range(n_zones):
                    zone_name = zone_names.get(z, f"Zone {z}")
                    legend_elements.append(
                        Patch(facecolor=zone_colors[z], edgecolor='none',
                              label=f"Zone {z}: {zone_name}")
                    )

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
                if method == 'xgboost':
                    model_display_name = "XGBoost"
                elif method == 'baggingpu_xgboost':
                    model_display_name = "BaggingPU(XGBoost)"
                else:
                    model_display_name = method
                ax.set_title(model_display_name, fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(False)

                plt.tight_layout()

                for fmt in formats:
                    fig.savefig(output_dir / f"fold_{fold_idx}_zone_map.{fmt}",
                                dpi=dpi, bbox_inches='tight', pad_inches=0.02)
                plt.close(fig)

    logger.info(f"\nFigures saved to: {base_output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot block examples")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['xgboost', 'baggingpu_xgboost'],
        help="Model method (default: first method in config)"
    )
    args = parser.parse_args()
    plot_block_examples(method=args.method)


if __name__ == "__main__":
    main()
