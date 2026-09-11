#!/usr/bin/env python3
"""15: Plot Zone Distribution"""

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

from scripts.utils import load_config, load_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_zone_distribution_for_mode(
    zone_mode: str,
    config: dict,
    base_output_dir: Path
):
    """Plot zone distribution analysis for a specific zone mode."""
    logger.info(f"\n--- Plotting Zone Distribution: {zone_mode.upper()} ---")

    output_dir = base_output_dir / zone_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = config['methods']
    dpi = config['visualization']['dpi']
    zone_colors = config['visualization']['colormap_zone']

    mode_title = "Absolute Prob" if zone_mode == "prob" else "Lift Percentile"

    for method in methods:
        logger.info(f"Processing {method}...")

        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method / 'practical_zone' / zone_mode
        results_path = agg_dir / "all_fold_results.csv"

        if not results_path.exists():
            logger.warning(f"Results not found for {method}/{zone_mode}: {results_path}")
            continue

        df = pd.read_csv(results_path)

        import re
        zone_pct_pattern = re.compile(r'^zone_(\d+)_pct$')
        zone_cols_available = [c for c in df.columns if zone_pct_pattern.match(c)]
        actual_n_zones = len(zone_cols_available)
        logger.info(f"Detected {actual_n_zones} zones in data")

        fig, ax = plt.subplots(figsize=config['visualization']['figsize_single'])

        zone_cols = [f'zone_{z}_pct' for z in range(actual_n_zones)]
        zone_data = df[zone_cols].values

        x = np.arange(len(df))
        bottom = np.zeros(len(df))

        for z in range(actual_n_zones):
            zone_def = config['zone_definitions'].get(z, {})
            zone_name = zone_def.get('name', f'Zone {z}')
            color = zone_colors[z] if z < len(zone_colors) else f'C{z}'
            ax.bar(x, zone_data[:, z], bottom=bottom, label=f'Z{z}: {zone_name}',
                   color=color, edgecolor='white', linewidth=0.5)
            bottom += zone_data[:, z]

        ax.set_xlabel('Outer Fold')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'Zone Distribution by Fold ({method.upper()}, {mode_title})')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in range(len(df))])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 100)

        fig.savefig(output_dir / f"zone_distribution_by_fold_{method}.png",
                    dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        deposit_cols = [f'zone_{z}_deposit_pct' for z in range(actual_n_zones)]
        if all(col in df.columns for col in deposit_cols):
            fig, ax = plt.subplots(figsize=config['visualization']['figsize_single'])

            deposit_data = [df[col].values for col in deposit_cols]

            bp = ax.boxplot(deposit_data, patch_artist=True)
            for i, (patch, col) in enumerate(zip(bp['boxes'], deposit_cols)):
                color = zone_colors[i] if i < len(zone_colors) else f'C{i}'
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel('Zone')
            ax.set_ylabel('Deposit Capture (%)')
            ax.set_title(f'Deposit Capture by Zone ({method.upper()}, {mode_title})')
            ax.set_xticklabels([f'Zone {z}' for z in range(actual_n_zones)])
            ax.grid(axis='y', alpha=0.3)

            fig.savefig(output_dir / f"deposit_capture_by_zone_{method}.png",
                        dpi=dpi, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, actual_n_zones * 1.5), 6))

        zone_pct_cols = [f'zone_{z}_pct' for z in range(actual_n_zones)]
        zone_pct_data = df[zone_pct_cols].values

        im = ax.imshow(zone_pct_data, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Fold')
        ax.set_xticks(range(actual_n_zones))
        ax.set_xticklabels([f'Zone {z}' for z in range(actual_n_zones)])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f'Fold {i}' for i in range(len(df))])
        ax.set_title(f'Zone Distribution Consistency ({method.upper()}, {mode_title})')

        for i in range(len(df)):
            for j in range(actual_n_zones):
                text = ax.text(j, i, f'{zone_pct_data[i, j]:.1f}%',
                               ha='center', va='center', fontsize=8)

        fig.colorbar(im, ax=ax, label='Percentage (%)')

        fig.savefig(output_dir / f"zone_consistency_heatmap_{method}.png",
                    dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

    logger.info(f"Figures saved to: {output_dir}")


def plot_zone_distribution(config: dict = None):
    """Plot zone distribution analysis for all zone modes."""
    logger.info("=" * 70)
    logger.info("Plotting Zone Distribution (Prob & Lift modes)")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    base_output_dir = PROJECT_ROOT / config['output_dirs']['figures'] / "zone_analysis"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])

    for zone_mode in zone_modes:
        plot_zone_distribution_for_mode(zone_mode, config, base_output_dir)

    logger.info(f"\nAll figures saved to: {base_output_dir}")


def main():
    plot_zone_distribution()


if __name__ == "__main__":
    main()
