#!/usr/bin/env python3
"""12: Aggregate Fold Results"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import (
    load_config,
    load_json,
    save_json
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

N_ZONES = 5


def aggregate_practical_zone_results(
    method: str,
    zone_mode: str,
    config: dict,
    output_dir: Path
) -> Optional[dict]:
    """Aggregate practical zone results for a specific zone mode."""
    logger.info(f"\n--- Aggregating Practical Zone: {zone_mode.upper()} ---")

    n_outer_folds = config['spatial_cv']['n_outer_folds']
    fold_results = []

    for fold_idx in range(n_outer_folds):
        pz_dir = (
            PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
            method / f'outer_fold_{fold_idx}' / 'practical_zone' / zone_mode
        )

        deposit_path = pz_dir / "deposit_analysis.json"
        if not deposit_path.exists():
            logger.warning(f"Practical zone data not found for fold {fold_idx}: {deposit_path}")
            continue

        deposit_analysis = load_json(deposit_path)

        fold_summary = {
            'fold': fold_idx,
            'zone_mode': zone_mode,
            'n_zones': deposit_analysis.get('n_zones', N_ZONES)
        }

        zone_analysis = deposit_analysis.get('zone_analysis', {})
        for z in range(N_ZONES):
            zone_key = f'zone_{z}'
            if zone_key in zone_analysis:
                fold_summary[f'{zone_key}_pct'] = zone_analysis[zone_key]['sample_pct']
                fold_summary[f'{zone_key}_deposit_pct'] = zone_analysis[zone_key]['deposit_pct']
                fold_summary[f'{zone_key}_cumulative_pct'] = zone_analysis[zone_key]['cumulative_deposit_pct']

        efficiency = deposit_analysis.get('efficiency_metrics', {})
        fold_summary['zone_0_1_capture'] = efficiency.get('zone_0_1_capture', 0)
        fold_summary['zone_0_1_2_capture'] = efficiency.get('zone_0_1_2_capture', 0)
        fold_summary['zone_4_miss_rate'] = efficiency.get('zone_4_miss_rate', 0)

        validation = deposit_analysis.get('validation', {})
        fold_summary['validation_z01_pass'] = validation.get('zone_0_1_capture_min_25', {}).get('pass', False)
        fold_summary['validation_z012_pass'] = validation.get('zone_0_1_2_capture_min_40', {}).get('pass', False)
        fold_summary['validation_z4_pass'] = validation.get('zone_4_capture_max_10', {}).get('pass', False)

        fold_results.append(fold_summary)
        logger.info(f"Fold {fold_idx}: Z0+1={fold_summary['zone_0_1_capture']:.1f}%, "
                   f"Z4 miss={fold_summary['zone_4_miss_rate']:.1f}%")

    if not fold_results:
        logger.warning(f"No practical zone results found for {zone_mode}")
        return None

    results_df = pd.DataFrame(fold_results)

    summary_stats = {
        'method': method,
        'zone_mode': zone_mode,
        'n_folds': len(fold_results),
        'n_zones': N_ZONES,
        'zone_distribution': {},
        'zone_capture': {},
        'efficiency_metrics': {
            'zone_0_1_capture': {
                'mean': float(results_df['zone_0_1_capture'].mean()),
                'std': float(results_df['zone_0_1_capture'].std())
            },
            'zone_0_1_2_capture': {
                'mean': float(results_df['zone_0_1_2_capture'].mean()),
                'std': float(results_df['zone_0_1_2_capture'].std())
            },
            'zone_4_miss_rate': {
                'mean': float(results_df['zone_4_miss_rate'].mean()),
                'std': float(results_df['zone_4_miss_rate'].std())
            }
        },
        'validation': {
            'z01_pass_rate': float(results_df['validation_z01_pass'].mean() * 100),
            'z012_pass_rate': float(results_df['validation_z012_pass'].mean() * 100),
            'z4_pass_rate': float(results_df['validation_z4_pass'].mean() * 100)
        }
    }

    for z in range(N_ZONES):
        pct_col = f'zone_{z}_pct'
        dep_col = f'zone_{z}_deposit_pct'
        cum_col = f'zone_{z}_cumulative_pct'

        if pct_col in results_df.columns:
            summary_stats['zone_distribution'][f'zone_{z}'] = {
                'mean': float(results_df[pct_col].mean()),
                'std': float(results_df[pct_col].std())
            }

        if dep_col in results_df.columns:
            summary_stats['zone_capture'][f'zone_{z}'] = {
                'mean': float(results_df[dep_col].mean()),
                'std': float(results_df[dep_col].std()),
                'cumulative_mean': float(results_df[cum_col].mean()),
                'cumulative_std': float(results_df[cum_col].std())
            }

    pz_output_dir = output_dir / 'practical_zone' / zone_mode
    pz_output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(pz_output_dir / "all_fold_results.csv", index=False)
    save_json(summary_stats, pz_output_dir / "summary_statistics.json")

    logger.info(f"Practical zone ({zone_mode}) results saved to: {pz_output_dir}")

    eff = summary_stats['efficiency_metrics']
    logger.info(f"  Z0+1 capture: {eff['zone_0_1_capture']['mean']:.1f}% ± {eff['zone_0_1_capture']['std']:.1f}%")
    logger.info(f"  Z0+1+2 capture: {eff['zone_0_1_2_capture']['mean']:.1f}% ± {eff['zone_0_1_2_capture']['std']:.1f}%")
    logger.info(f"  Z4 miss rate: {eff['zone_4_miss_rate']['mean']:.1f}% ± {eff['zone_4_miss_rate']['std']:.1f}%")

    return summary_stats


def aggregate_fold_results(
    method: str = None,
    config: dict = None
):
    """Aggregate results from all outer folds."""
    logger.info("=" * 70)
    logger.info("Aggregating Fold Results")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    methods = [method] if method else config['methods']
    n_outer_folds = config['spatial_cv']['n_outer_folds']
    zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])

    for m in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Method: {m.upper()}")
        logger.info(f"{'='*50}")

        fold_results = []
        merged_predictions = []

        for fold_idx in range(n_outer_folds):
            eval_dir = (
                PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
                m / f'outer_fold_{fold_idx}'
            )

            metrics_path = eval_dir / "metrics.json"
            if not metrics_path.exists():
                logger.warning(f"Metrics not found for fold {fold_idx}: {metrics_path}")
                continue

            metrics = load_json(metrics_path)

            deposit_path = eval_dir / "deposit_analysis.json"
            if deposit_path.exists():
                deposit_analysis = load_json(deposit_path)
            else:
                deposit_analysis = None
                logger.warning(f"Deposit analysis not found for fold {fold_idx}")

            fold_summary = {
                'fold': fold_idx,
                'n_test': metrics['n_test'],
                'n_deposits': metrics['n_deposits'],
                'coverage': metrics['coverage']['class_1_coverage'],
                'fnr': metrics['coverage']['fnr'],
                'pr_auc': metrics['pr_auc'],
                'threshold': metrics['calibration']['threshold'],
                'platt_coef': metrics['calibration']['platt_coef'],
                'platt_intercept': metrics['calibration']['platt_intercept']
            }

            n_zones = metrics.get('zone_distribution', {}).get('n_zones', N_ZONES)
            if deposit_analysis:
                n_zones = deposit_analysis.get('zone_analysis', {}).get('n_zones', n_zones)
            fold_summary['n_zones'] = n_zones

            for z in range(n_zones):
                zone_key = f'zone_{z}'
                zone_dist = metrics.get('zone_distribution', {})
                if zone_key in zone_dist:
                    fold_summary[f'{zone_key}_pct'] = zone_dist[zone_key].get('pct', 0)
                if deposit_analysis:
                    zone_ana = deposit_analysis.get('zone_analysis', {})
                    if zone_key in zone_ana:
                        fold_summary[f'{zone_key}_deposit_pct'] = zone_ana[zone_key]['deposit_pct']
                        fold_summary[f'{zone_key}_cumulative_pct'] = zone_ana[zone_key]['cumulative_deposit_pct']

            fold_results.append(fold_summary)

            predictions_path = eval_dir / "test_predictions.csv"
            if predictions_path.exists():
                pred_df = pd.read_csv(predictions_path)
                pred_df['fold'] = fold_idx
                merged_predictions.append(pred_df)

            logger.info(f"Fold {fold_idx}: coverage={fold_summary['coverage']:.4f}, "
                        f"PR-AUC={fold_summary['pr_auc']:.4f}")

        if not fold_results:
            logger.error(f"No results found for method {m}")
            continue

        results_df = pd.DataFrame(fold_results)

        summary_stats = {
            'method': m,
            'n_folds': len(fold_results),
            'coverage': {
                'mean': float(results_df['coverage'].mean()),
                'std': float(results_df['coverage'].std()),
                'min': float(results_df['coverage'].min()),
                'max': float(results_df['coverage'].max())
            },
            'fnr': {
                'mean': float(results_df['fnr'].mean()),
                'std': float(results_df['fnr'].std()),
                'min': float(results_df['fnr'].min()),
                'max': float(results_df['fnr'].max())
            },
            'pr_auc': {
                'mean': float(results_df['pr_auc'].mean()),
                'std': float(results_df['pr_auc'].std()),
                'min': float(results_df['pr_auc'].min()),
                'max': float(results_df['pr_auc'].max())
            },
            'threshold': {
                'mean': float(results_df['threshold'].mean()),
                'std': float(results_df['threshold'].std()),
                'min': float(results_df['threshold'].min()),
                'max': float(results_df['threshold'].max())
            }
        }

        n_zones = int(results_df['n_zones'].max()) if 'n_zones' in results_df.columns else N_ZONES
        summary_stats['n_zones'] = n_zones
        summary_stats['zone_distribution'] = {}
        summary_stats['zone_capture'] = {}

        for z in range(n_zones):
            zone_pct_col = f'zone_{z}_pct'
            zone_deposit_col = f'zone_{z}_deposit_pct'
            zone_cumulative_col = f'zone_{z}_cumulative_pct'

            if zone_pct_col in results_df.columns:
                summary_stats['zone_distribution'][f'zone_{z}'] = {
                    'mean': float(results_df[zone_pct_col].mean()),
                    'std': float(results_df[zone_pct_col].std())
                }

            if zone_deposit_col in results_df.columns:
                summary_stats['zone_capture'][f'zone_{z}'] = {
                    'mean': float(results_df[zone_deposit_col].mean()),
                    'std': float(results_df[zone_deposit_col].std()),
                    'cumulative_mean': float(results_df[zone_cumulative_col].mean()),
                    'cumulative_std': float(results_df[zone_cumulative_col].std())
                }

        logger.info(f"\nAggregate Statistics:")
        logger.info(f"  Coverage: {summary_stats['coverage']['mean']:.4f} ± {summary_stats['coverage']['std']:.4f}")
        logger.info(f"  FNR: {summary_stats['fnr']['mean']:.4f} ± {summary_stats['fnr']['std']:.4f}")
        logger.info(f"  PR-AUC: {summary_stats['pr_auc']['mean']:.4f} ± {summary_stats['pr_auc']['std']:.4f}")

        if summary_stats['zone_capture']:
            logger.info(f"\nZone Capture (Deposit %, {n_zones} zones):")
            for z in range(n_zones):
                zc = summary_stats['zone_capture'].get(f'zone_{z}', {})
                if zc:
                    logger.info(f"  Zone {z}: {zc['mean']:.1f}% ± {zc['std']:.1f}%, "
                                f"cumulative: {zc['cumulative_mean']:.1f}%")

        output_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / m
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_dir / "all_fold_results.csv", index=False)

        save_json(summary_stats, output_dir / "summary_statistics.json")

        if merged_predictions:
            merged_df = pd.concat(merged_predictions, ignore_index=True)
            merged_df.to_csv(output_dir / "merged_predictions.csv", index=False)
            logger.info(f"Merged predictions: {len(merged_df):,} samples "
                        f"(should cover entire dataset)")

        logger.info(f"\nBase results saved to: {output_dir}")

        for zone_mode in zone_modes:
            aggregate_practical_zone_results(m, zone_mode, config, output_dir)

    logger.info("\nFold aggregation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from all outer folds"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['xgboost', 'baggingpu_xgboost', 'tabpfn'],
        help="Model method (default: all methods)"
    )

    args = parser.parse_args()
    aggregate_fold_results(args.method)


if __name__ == "__main__":
    main()
