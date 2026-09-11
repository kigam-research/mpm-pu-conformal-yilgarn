#!/usr/bin/env python3
"""11: Compute Deposit Capture Analysis"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import (
    load_config,
    save_json
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_deposit_capture(
    outer_fold: int,
    method: str = None,
    config: dict = None
):
    """Compute deposit capture analysis for test fold."""
    logger.info("=" * 70)
    logger.info(f"Deposit Capture Analysis - Outer Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    methods = [method] if method else config['methods']

    for m in methods:
        logger.info(f"\n--- Method: {m.upper()} ---")

        eval_dir = (
            PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
            m / f'outer_fold_{outer_fold}'
        )
        predictions_path = eval_dir / "test_predictions.csv"

        if not predictions_path.exists():
            logger.warning(f"Test predictions not found: {predictions_path}")
            logger.warning("Run evaluate_test_fold.py first!")
            continue

        predictions_df = pd.read_csv(predictions_path)

        target = predictions_df['target'].values
        zones = predictions_df['zone'].values
        calibrated_prob = predictions_df['calibrated_prob'].values
        bootstrap_std = predictions_df['bootstrap_std'].values

        if 'rel_iqr' in predictions_df.columns:
            rel_iqr = predictions_df['rel_iqr'].values
        else:
            rel_iqr = None

        total_samples = len(target)
        total_deposits = target.sum()

        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Total deposits: {total_deposits}")

        n_zones = len(config['zone_definitions'])
        zone_analysis = {'n_zones': n_zones}
        cumulative_deposits = 0
        cumulative_samples = 0

        for z in range(n_zones):
            zone_mask = zones == z
            n_samples = zone_mask.sum()
            n_deposits = target[zone_mask].sum()

            cumulative_deposits += n_deposits
            cumulative_samples += n_samples

            zone_probs = calibrated_prob[zone_mask]
            zone_stds = bootstrap_std[zone_mask]

            zone_stats = {
                'n_samples': int(n_samples),
                'n_deposits': int(n_deposits),
                'sample_pct': float(n_samples / total_samples * 100),
                'deposit_pct': float(n_deposits / total_deposits * 100) if total_deposits > 0 else 0,
                'cumulative_samples': int(cumulative_samples),
                'cumulative_deposits': int(cumulative_deposits),
                'cumulative_sample_pct': float(cumulative_samples / total_samples * 100),
                'cumulative_deposit_pct': float(cumulative_deposits / total_deposits * 100) if total_deposits > 0 else 0,
                'density_ratio': float((n_deposits / n_samples) / (total_deposits / total_samples)) if n_samples > 0 and total_deposits > 0 else 0,
                'prob_stats': {
                    'mean': float(zone_probs.mean()) if n_samples > 0 else 0,
                    'std': float(zone_probs.std()) if n_samples > 0 else 0,
                    'min': float(zone_probs.min()) if n_samples > 0 else 0,
                    'max': float(zone_probs.max()) if n_samples > 0 else 0
                },
                'uncertainty_stats': {
                    'mean': float(zone_stds.mean()) if n_samples > 0 else 0,
                    'std': float(zone_stds.std()) if n_samples > 0 else 0,
                    'min': float(zone_stds.min()) if n_samples > 0 else 0,
                    'max': float(zone_stds.max()) if n_samples > 0 else 0
                }
            }

            if rel_iqr is not None:
                zone_rel_iqr = rel_iqr[zone_mask]
                zone_stats['rel_iqr_stats'] = {
                    'mean': float(zone_rel_iqr.mean()) if n_samples > 0 else 0,
                    'std': float(zone_rel_iqr.std()) if n_samples > 0 else 0,
                    'min': float(zone_rel_iqr.min()) if n_samples > 0 else 0,
                    'max': float(zone_rel_iqr.max()) if n_samples > 0 else 0,
                    'pct_below_30': float((zone_rel_iqr <= 0.30).mean() * 100) if n_samples > 0 else 0
                }

            zone_analysis[f'zone_{z}'] = zone_stats

            zone_name = config['zone_definitions'][z]['name']
            logger.info(
                f"Zone {z} ({zone_name}): "
                f"{n_samples:,} samples ({zone_analysis[f'zone_{z}']['sample_pct']:.1f}%), "
                f"{n_deposits} deposits ({zone_analysis[f'zone_{z}']['deposit_pct']:.1f}%), "
                f"cumulative: {zone_analysis[f'zone_{z}']['cumulative_deposit_pct']:.1f}%"
            )


        cumulative_sample_fracs = []
        cumulative_deposit_fracs = []
        for z in range(n_zones):
            cumulative_sample_fracs.append(zone_analysis[f'zone_{z}']['cumulative_sample_pct'] / 100)
            cumulative_deposit_fracs.append(zone_analysis[f'zone_{z}']['cumulative_deposit_pct'] / 100)

        sample_fracs = [0] + cumulative_sample_fracs
        deposit_fracs = [0] + cumulative_deposit_fracs

        auc = 0
        for i in range(len(sample_fracs) - 1):
            auc += (sample_fracs[i+1] - sample_fracs[i]) * (deposit_fracs[i+1] + deposit_fracs[i]) / 2

        gini = 2 * auc - 1

        efficiency_metrics = {
            'deposit_capture_auc': float(auc),
            'gini_coefficient': float(gini),
            'zone_0_1_capture': float(zone_analysis['zone_1']['cumulative_deposit_pct']),
            'zone_0_1_2_capture': float(zone_analysis['zone_2']['cumulative_deposit_pct']),
            'zone_0_to_3_capture': float(zone_analysis['zone_3']['cumulative_deposit_pct']),
            'zone_4_miss_rate': float(zone_analysis['zone_4']['deposit_pct'])
        }

        if n_zones > 5 and 'zone_5' in zone_analysis:
            efficiency_metrics['zone_5_data_gap'] = float(zone_analysis['zone_5']['deposit_pct'])

        logger.info(f"\nEfficiency Metrics:")
        logger.info(f"  Deposit capture AUC: {efficiency_metrics['deposit_capture_auc']:.4f}")
        logger.info(f"  Gini coefficient: {efficiency_metrics['gini_coefficient']:.4f}")
        logger.info(f"  Zone 0-1 capture: {efficiency_metrics['zone_0_1_capture']:.1f}%")
        logger.info(f"  Zone 0-2 capture: {efficiency_metrics['zone_0_1_2_capture']:.1f}%")
        logger.info(f"  Zone 0-3 capture: {efficiency_metrics['zone_0_to_3_capture']:.1f}%")
        logger.info(f"  Zone 4 miss rate: {efficiency_metrics['zone_4_miss_rate']:.1f}%")
        if 'zone_5_data_gap' in efficiency_metrics:
            logger.info(f"  Zone 5 data gap: {efficiency_metrics['zone_5_data_gap']:.1f}%")

        validation = {
            'coverage_min_85': {
                'description': 'min(coverage) >= 85%',
                'threshold': config['validation_criteria']['coverage_min'] * 100,
                'value': None,
                'pass': None
            },
            'zone_0_1_capture_min_25': {
                'description': 'Zone 0+1 (high efficiency) deposit capture >= 25%',
                'threshold': config['validation_criteria']['zone_0_1_capture_min'] * 100,
                'value': zone_analysis['zone_1']['cumulative_deposit_pct'],
                'pass': zone_analysis['zone_1']['cumulative_deposit_pct'] >= config['validation_criteria']['zone_0_1_capture_min'] * 100
            },
            'zone_0_1_2_capture_min_40': {
                'description': 'Zone 0+1+2 (density>1.5x) deposit capture >= 40%',
                'threshold': config['validation_criteria']['zone_0_1_2_capture_min'] * 100,
                'value': zone_analysis['zone_2']['cumulative_deposit_pct'],
                'pass': zone_analysis['zone_2']['cumulative_deposit_pct'] >= config['validation_criteria']['zone_0_1_2_capture_min'] * 100
            },
            'zone_4_capture_max_10': {
                'description': 'Zone 4 (excluded) deposit miss < 10%',
                'threshold': config['validation_criteria']['zone_4_capture_max'] * 100,
                'value': zone_analysis['zone_4']['deposit_pct'],
                'pass': zone_analysis['zone_4']['deposit_pct'] < config['validation_criteria']['zone_4_capture_max'] * 100
            }
        }

        metrics_path = eval_dir / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            coverage = metrics['coverage']['class_1_coverage'] * 100
            validation['coverage_min_85']['value'] = coverage
            validation['coverage_min_85']['pass'] = coverage >= config['validation_criteria']['coverage_min'] * 100

        logger.info(f"\nValidation Criteria (Option C):")
        for criterion, v in validation.items():
            status = "PASS" if v['pass'] else "FAIL" if v['pass'] is not None else "N/A"
            value_str = f"{v['value']:.1f}%" if v['value'] is not None else "N/A"
            threshold_str = f"{v['threshold']:.0f}%" if 'max' not in criterion else f"<{v['threshold']:.0f}%"
            logger.info(f"  {criterion}: {value_str} (threshold: {threshold_str}) [{status}]")

        deposit_analysis = {
            'outer_fold': outer_fold,
            'method': m,
            'total_samples': int(total_samples),
            'total_deposits': int(total_deposits),
            'zone_analysis': zone_analysis,
            'efficiency_metrics': efficiency_metrics,
            'validation': validation
        }

        save_json(deposit_analysis, eval_dir / "deposit_analysis.json")
        logger.info(f"\nResults saved to: {eval_dir / 'deposit_analysis.json'}")

    logger.info("\nDeposit capture analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Compute deposit capture analysis"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index (0-4)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['xgboost', 'baggingpu_xgboost', 'tabpfn'],
        help="Model method (default: all methods)"
    )

    args = parser.parse_args()
    compute_deposit_capture(args.outer_fold, args.method)


if __name__ == "__main__":
    main()
