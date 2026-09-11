#!/usr/bin/env python3
"""14: Plot CV Metrics"""

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


def plot_cv_metrics(config: dict = None):
    """Plot CV metrics comparison across folds."""
    logger.info("=" * 70)
    logger.info("Plotting CV Metrics")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    output_dir = PROJECT_ROOT / config['output_dirs']['figures'] / "cv_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = config['methods']
    n_folds = config['spatial_cv']['n_outer_folds']
    formats = config['visualization']['figure_formats']
    dpi = config['visualization']['dpi']

    method_results = {}
    for method in methods:
        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / method
        results_path = agg_dir / "all_fold_results.csv"
        if results_path.exists():
            method_results[method] = pd.read_csv(results_path)
        else:
            logger.warning(f"Results not found for {method}")

    if not method_results:
        logger.error("No results found to plot")
        return

    fig, ax = plt.subplots(figsize=config['visualization']['figsize_single'])

    x = np.arange(n_folds)
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e']

    for i, (method, df) in enumerate(method_results.items()):
        offset = (i - len(method_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df['pr_auc'], width, label=method.upper(), color=colors[i % 2])

    ax.set_xlabel('Outer Fold')
    ax.set_ylabel('PR-AUC')
    ax.set_title('PR-AUC by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    for fmt in formats:
        fig.savefig(output_dir / f"fold_comparison_prauc.{fmt}", dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved PR-AUC comparison plot")

    fig, ax = plt.subplots(figsize=config['visualization']['figsize_single'])

    for i, (method, df) in enumerate(method_results.items()):
        offset = (i - len(method_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df['coverage'] * 100, width, label=method.upper(), color=colors[i % 2])

    ax.axhline(y=85, color='red', linestyle='--', label='Min threshold (85%)')
    ax.set_xlabel('Outer Fold')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Class 1 Coverage by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    for fmt in formats:
        fig.savefig(output_dir / f"fold_comparison_coverage.{fmt}", dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved Coverage comparison plot")

    fig, ax = plt.subplots(figsize=config['visualization']['figsize_single'])

    for i, (method, df) in enumerate(method_results.items()):
        offset = (i - len(method_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df['fnr'] * 100, width, label=method.upper(), color=colors[i % 2])

    ax.axhline(y=15, color='red', linestyle='--', label='Max threshold (15%)')
    ax.set_xlabel('Outer Fold')
    ax.set_ylabel('FNR (%)')
    ax.set_title('False Negative Rate by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in range(n_folds)])
    ax.legend()
    ax.set_ylim(0, 30)
    ax.grid(axis='y', alpha=0.3)

    for fmt in formats:
        fig.savefig(output_dir / f"fold_comparison_fnr.{fmt}", dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved FNR comparison plot")

    logger.info(f"\nFigures saved to: {output_dir}")


def main():
    plot_cv_metrics()


if __name__ == "__main__":
    main()
