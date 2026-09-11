#!/usr/bin/env python3
"""Aggregate OOF Conformal Results"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import load_config, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

N_ZONES = 5


def aggregate_practical_zone_results_oof(
    method: str,
    zone_mode: str,
    config: dict,
    output_dir: Path
) -> Optional[dict]:
    """Aggregate practical zone results for OOF conformal."""
    logger.info(f"\n--- [OOF] Aggregating Practical Zone: {zone_mode.upper()} ---")

    n_outer_folds = config['spatial_cv']['n_outer_folds']
    fold_results = []

    for fold_idx in range(n_outer_folds):
        pz_dir = (
            PROJECT_ROOT / 'outputs' / 'test_evaluation_oof' /
            method / f'outer_fold_{fold_idx}' / 'practical_zone' / zone_mode
        )

        deposit_path = pz_dir / "deposit_analysis.json"
        if not deposit_path.exists():
            logger.warning(f"OOF practical zone data not found for fold {fold_idx}")
            continue

        deposit_analysis = load_json(deposit_path)

        fold_summary = {
            'fold': fold_idx,
            'zone_mode': zone_mode,
            'conformal_method': 'oof_pooling',
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

    if not fold_results:
        return None

    results_df = pd.DataFrame(fold_results)

    summary_stats = {
        'method': method,
        'zone_mode': zone_mode,
        'conformal_method': 'oof_pooling',
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

    eff = summary_stats['efficiency_metrics']
    logger.info(f"  Z0+1: {eff['zone_0_1_capture']['mean']:.1f}% +/- {eff['zone_0_1_capture']['std']:.1f}%")
    logger.info(f"  Z0+1+2: {eff['zone_0_1_2_capture']['mean']:.1f}% +/- {eff['zone_0_1_2_capture']['std']:.1f}%")
    logger.info(f"  Z4 miss: {eff['zone_4_miss_rate']['mean']:.1f}% +/- {eff['zone_4_miss_rate']['std']:.1f}%")

    return summary_stats


def generate_comparison(method, oof_summary, config, output_dir):
    """Compare OOF results with original 20-iteration results."""
    logger.info(f"\n--- Comparison: OOF vs Original (20-iteration) ---")

    original_summary_path = (
        PROJECT_ROOT / config['output_dirs']['aggregated'] /
        method / "summary_statistics.json"
    )

    if not original_summary_path.exists():
        logger.warning(f"Original summary not found: {original_summary_path}")
        return None

    original_summary = load_json(original_summary_path)

    comparison = {
        'method': method,
        'original_method': '20_iteration_median',
        'oof_method': 'oof_pooling',
        'coverage': {
            'original': original_summary['coverage'],
            'oof': oof_summary['coverage'],
            'diff_mean': oof_summary['coverage']['mean'] - original_summary['coverage']['mean']
        },
        'fnr': {
            'original': original_summary['fnr'],
            'oof': oof_summary['fnr'],
            'diff_mean': oof_summary['fnr']['mean'] - original_summary['fnr']['mean']
        },
        'pr_auc': {
            'original': original_summary['pr_auc'],
            'oof': oof_summary['pr_auc'],
            'diff_mean': oof_summary['pr_auc']['mean'] - original_summary['pr_auc']['mean']
        },
        'threshold': {
            'original': original_summary['threshold'],
            'oof': oof_summary['threshold'],
            'diff_mean': oof_summary['threshold']['mean'] - original_summary['threshold']['mean']
        }
    }

    if original_summary.get('zone_capture') and oof_summary.get('zone_capture'):
        comparison['zone_capture'] = {}
        for z in range(N_ZONES):
            zk = f'zone_{z}'
            if zk in original_summary['zone_capture'] and zk in oof_summary['zone_capture']:
                comparison['zone_capture'][zk] = {
                    'original_mean': original_summary['zone_capture'][zk]['mean'],
                    'oof_mean': oof_summary['zone_capture'][zk]['mean'],
                    'diff': oof_summary['zone_capture'][zk]['mean'] - original_summary['zone_capture'][zk]['mean']
                }

    save_json(comparison, output_dir / "comparison_with_original.json")

    logger.info(f"\n{'Metric':<20} {'Original':>12} {'OOF':>12} {'Diff':>10}")
    logger.info("-" * 56)
    logger.info(f"{'Coverage':<20} {original_summary['coverage']['mean']:>11.4f} "
                f"{oof_summary['coverage']['mean']:>11.4f} "
                f"{comparison['coverage']['diff_mean']:>+9.4f}")
    logger.info(f"{'FNR':<20} {original_summary['fnr']['mean']:>11.4f} "
                f"{oof_summary['fnr']['mean']:>11.4f} "
                f"{comparison['fnr']['diff_mean']:>+9.4f}")
    logger.info(f"{'PR-AUC':<20} {original_summary['pr_auc']['mean']:>11.4f} "
                f"{oof_summary['pr_auc']['mean']:>11.4f} "
                f"{comparison['pr_auc']['diff_mean']:>+9.4f}")
    logger.info(f"{'Threshold':<20} {original_summary['threshold']['mean']:>11.6f} "
                f"{oof_summary['threshold']['mean']:>11.6f} "
                f"{comparison['threshold']['diff_mean']:>+9.6f}")

    return comparison


def aggregate_oof_results(method=None, config=None):
    """Aggregate OOF results from all outer folds."""
    logger.info("=" * 70)
    logger.info("[OOF] Aggregating Fold Results")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    methods = [method] if method else ['xgboost', 'baggingpu_xgboost']
    n_outer_folds = config['spatial_cv']['n_outer_folds']
    zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])

    for m in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Method: {m.upper()} (OOF Pooling)")
        logger.info(f"{'='*50}")

        fold_results = []
        merged_predictions = []

        for fold_idx in range(n_outer_folds):
            eval_dir = (
                PROJECT_ROOT / 'outputs' / 'test_evaluation_oof' /
                m / f'outer_fold_{fold_idx}'
            )

            metrics_path = eval_dir / "metrics.json"
            if not metrics_path.exists():
                logger.warning(f"OOF metrics not found for fold {fold_idx}")
                continue

            metrics = load_json(metrics_path)

            fold_summary = {
                'fold': fold_idx,
                'conformal_method': 'oof_pooling',
                'n_test': metrics['n_test'],
                'n_deposits': metrics['n_deposits'],
                'coverage': metrics['coverage']['class_1_coverage'],
                'fnr': metrics['coverage']['fnr'],
                'pr_auc': metrics['pr_auc'],
                'threshold': metrics['calibration']['threshold'],
                'platt_coef': metrics['calibration']['platt_coef'],
                'platt_intercept': metrics['calibration']['platt_intercept'],
                'n_positives_calibration': metrics['calibration'].get('n_positives_calibration'),
                'hoeffding_correction': metrics['calibration'].get('hoeffding_correction')
            }

            n_zones = metrics.get('zone_distribution', {}).get('n_zones', N_ZONES)
            fold_summary['n_zones'] = n_zones

            zone_capture = metrics.get('zone_capture', {})
            for z in range(n_zones):
                zone_key = f'zone_{z}'
                zone_dist = metrics.get('zone_distribution', {})
                if zone_key in zone_dist:
                    fold_summary[f'{zone_key}_pct'] = zone_dist[zone_key].get('pct', 0)
                if zone_key in zone_capture:
                    fold_summary[f'{zone_key}_deposit_pct'] = zone_capture[zone_key].get('deposit_pct', 0)
                    fold_summary[f'{zone_key}_cumulative_pct'] = zone_capture[zone_key].get('cumulative_deposit_pct', 0)

            fold_results.append(fold_summary)

            predictions_path = eval_dir / "test_predictions.csv"
            if predictions_path.exists():
                pred_df = pd.read_csv(predictions_path)
                pred_df['fold'] = fold_idx
                merged_predictions.append(pred_df)

            logger.info(f"Fold {fold_idx}: coverage={fold_summary['coverage']:.4f}, "
                        f"PR-AUC={fold_summary['pr_auc']:.4f}, "
                        f"threshold={fold_summary['threshold']:.6f}")

        if not fold_results:
            logger.error(f"No OOF results found for method {m}")
            continue

        results_df = pd.DataFrame(fold_results)

        summary_stats = {
            'method': m,
            'conformal_method': 'oof_pooling',
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

        if 'n_positives_calibration' in results_df.columns:
            summary_stats['calibration'] = {
                'n_positives_calibration': {
                    'mean': float(results_df['n_positives_calibration'].mean()),
                    'std': float(results_df['n_positives_calibration'].std())
                },
                'hoeffding_correction': {
                    'mean': float(results_df['hoeffding_correction'].mean()),
                    'std': float(results_df['hoeffding_correction'].std())
                }
            }

        n_zones = int(results_df['n_zones'].max()) if 'n_zones' in results_df.columns else N_ZONES
        summary_stats['n_zones'] = n_zones
        summary_stats['zone_distribution'] = {}
        summary_stats['zone_capture'] = {}

        for z in range(n_zones):
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

        logger.info(f"\nAggregate Statistics (OOF):")
        logger.info(f"  Coverage: {summary_stats['coverage']['mean']:.4f} +/- {summary_stats['coverage']['std']:.4f}")
        logger.info(f"  FNR: {summary_stats['fnr']['mean']:.4f} +/- {summary_stats['fnr']['std']:.4f}")
        logger.info(f"  PR-AUC: {summary_stats['pr_auc']['mean']:.4f} +/- {summary_stats['pr_auc']['std']:.4f}")
        logger.info(f"  Threshold: {summary_stats['threshold']['mean']:.6f} +/- {summary_stats['threshold']['std']:.6f}")

        output_dir = PROJECT_ROOT / 'outputs' / 'aggregated_oof' / m
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_dir / "all_fold_results.csv", index=False)
        save_json(summary_stats, output_dir / "summary_statistics.json")

        if merged_predictions:
            merged_df = pd.concat(merged_predictions, ignore_index=True)
            merged_df.to_csv(output_dir / "merged_predictions.csv", index=False)
            logger.info(f"Merged predictions: {len(merged_df):,} samples")

        for zone_mode in zone_modes:
            aggregate_practical_zone_results_oof(m, zone_mode, config, output_dir)

        generate_comparison(m, summary_stats, config, output_dir)

        logger.info(f"\nOOF results saved to: {output_dir}")

    logger.info("\n[OOF] Fold aggregation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate OOF conformal results"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=['xgboost', 'baggingpu_xgboost'],
        help="Model method (default: all)"
    )
    args = parser.parse_args()
    aggregate_oof_results(args.method)


if __name__ == "__main__":
    main()
