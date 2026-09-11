#!/usr/bin/env python3
"""16: Plot Deposit Capture Analysis"""

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

N_ZONES = 5


def plot_deposit_capture_for_mode(
    zone_mode: str,
    config: dict,
    base_output_dir: Path
):
    """Plot deposit capture analysis for a specific zone mode."""
    logger.info(f"\n--- Plotting Deposit Capture: {zone_mode.upper()} ---")

    output_dir = base_output_dir / zone_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = config['methods']
    dpi = config['visualization']['dpi']
    zone_colors = config['visualization']['colormap_zone']

    mode_title = "Absolute Prob" if zone_mode == "prob" else "Lift Percentile"

    method_curves = {}

    for method in methods:
        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method / 'practical_zone' / zone_mode
        results_path = agg_dir / "all_fold_results.csv"

        if not results_path.exists():
            logger.warning(f"Results not found for {method}/{zone_mode}")
            continue

        df = pd.read_csv(results_path)

        cumulative_cols = [f'zone_{z}_cumulative_pct' for z in range(N_ZONES)
                          if f'zone_{z}_cumulative_pct' in df.columns]
        actual_n_zones = len(cumulative_cols)

        if actual_n_zones > 0:
            mean_cumulative = df[cumulative_cols].mean().values
            std_cumulative = df[cumulative_cols].std().values
            method_curves[method] = (mean_cumulative, std_cumulative, actual_n_zones)

    if not method_curves:
        logger.error(f"No cumulative data found for {zone_mode}")
        return

    max_zones = max(data[2] for data in method_curves.values())

    fig, ax = plt.subplots(figsize=config['visualization']['figsize_single'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    x = np.arange(max_zones)

    for i, (method, (mean_cum, std_cum, n_z)) in enumerate(method_curves.items()):
        x_method = np.arange(n_z)
        ax.plot(x_method, mean_cum, 'o-', color=colors[i % len(colors)],
                label=method.upper(), linewidth=2, markersize=8)
        ax.fill_between(x_method, mean_cum - std_cum, mean_cum + std_cum,
                        color=colors[i % len(colors)], alpha=0.2)

    ax.plot(x, np.linspace(0, 100, max_zones), 'k--', label='Random', alpha=0.5)

    ax.axhline(y=90, color='green', linestyle=':', label='Target (90%)', alpha=0.7)
    ax.axhline(y=40, color='orange', linestyle=':', label='Zone 0-2 Target (40%)', alpha=0.7)

    ax.set_xlabel('Zone (Cumulative)')
    ax.set_ylabel('Deposit Capture (%)')
    ax.set_title(f'Cumulative Deposit Capture by Zone ({mode_title})\n'
                 f'(Z0-2: Exploration, Z3: Potential, Z4: Excluded)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Zone 0-{z}' for z in range(max_zones)])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    fig.savefig(output_dir / "deposit_capture_curve.png", dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved deposit capture curve")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    for i, method in enumerate(methods):
        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method / 'practical_zone' / zone_mode
        results_path = agg_dir / "all_fold_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            if 'zone_1_cumulative_pct' in df.columns:
                mean_val = df['zone_1_cumulative_pct'].mean()
                std_val = df['zone_1_cumulative_pct'].std()
                ax.bar(i, mean_val, yerr=std_val, capsize=5,
                       color=colors[i % len(colors)], label=method.upper())

    ax.axhline(y=25, color='red', linestyle='--', label='Min threshold (25%)')
    ax.set_ylabel('Deposit Capture (%)')
    ax.set_title(f'Zone 0+1 (High Efficiency) Capture\n({mode_title})')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    for i, method in enumerate(methods):
        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method / 'practical_zone' / zone_mode
        results_path = agg_dir / "all_fold_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            if 'zone_2_cumulative_pct' in df.columns:
                mean_val = df['zone_2_cumulative_pct'].mean()
                std_val = df['zone_2_cumulative_pct'].std()
                ax.bar(i, mean_val, yerr=std_val, capsize=5,
                       color=colors[i % len(colors)], label=method.upper())

    ax.axhline(y=40, color='red', linestyle='--', label='Min threshold (40%)')
    ax.set_ylabel('Deposit Capture (%)')
    ax.set_title(f'Zone 0-2 (Exploration) Capture\n({mode_title})')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    for i, method in enumerate(methods):
        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method / 'practical_zone' / zone_mode
        results_path = agg_dir / "all_fold_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            if 'zone_4_deposit_pct' in df.columns:
                mean_val = df['zone_4_deposit_pct'].mean()
                std_val = df['zone_4_deposit_pct'].std()
                ax.bar(i, mean_val, yerr=std_val, capsize=5,
                       color=colors[i % len(colors)], label=method.upper())

    ax.axhline(y=10, color='red', linestyle='--', label='Max threshold (10%)')
    ax.set_ylabel('Deposit Capture (%)')
    ax.set_title(f'Zone 4 (EXCLUDED) Miss Rate\n({mode_title})')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "method_comparison_capture.png", dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved method comparison plot")

    logger.info(f"Figures saved to: {output_dir}")


def plot_deposit_capture(config: dict = None):
    """Plot deposit capture analysis for all zone modes."""
    logger.info("=" * 70)
    logger.info("Plotting Deposit Capture Analysis (Prob & Lift modes)")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    base_output_dir = PROJECT_ROOT / config['output_dirs']['figures'] / "zone_analysis"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])

    for zone_mode in zone_modes:
        plot_deposit_capture_for_mode(zone_mode, config, base_output_dir)

    logger.info(f"\nAll figures saved to: {base_output_dir}")


def main():
    plot_deposit_capture()


if __name__ == "__main__":
    main()
