#!/usr/bin/env python3
"""Evaluate Test Fold using OOF Pooling Conformal Parameters"""

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
    load_original_data,
    load_outer_folds,
    load_json,
    save_json,
    PlattApplicator,
    compute_prediction_sets,
    compute_fnr_practical_zones,
    compute_coverage_metrics,
    compute_deposit_capture_by_zone,
    compute_pr_auc
)
from scripts.utils.conformal_utils import assign_zones_jorc30

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_test_fold_oof(
    outer_fold: int,
    method: str,
    config: dict = None
):
    """Evaluate model on outer test fold using OOF conformal params."""
    logger.info("=" * 70)
    logger.info(f"[OOF] Test Fold Evaluation - {method.upper()} - Outer Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    bootstrap_dir = (
        PROJECT_ROOT / config['output_dirs']['models'] /
        method / f'outer_fold_{outer_fold}'
    )
    bootstrap_pred_path = bootstrap_dir / "bootstrap_predictions.npz"

    if not bootstrap_pred_path.exists():
        logger.error(f"Bootstrap predictions not found: {bootstrap_pred_path}")
        sys.exit(1)

    logger.info(f"Loading bootstrap predictions from: {bootstrap_pred_path}")
    bootstrap_data = np.load(bootstrap_pred_path)

    test_indices = bootstrap_data['test_indices']
    test_mean = bootstrap_data['test_mean']
    test_std = bootstrap_data['test_std']
    y_test = bootstrap_data['y_test']

    test_predictions_raw = bootstrap_data.get('test_predictions', None)
    if test_predictions_raw is not None:
        q25 = np.percentile(test_predictions_raw, 25, axis=0)
        q75 = np.percentile(test_predictions_raw, 75, axis=0)
        logger.info(f"Computed q25/q75 from {test_predictions_raw.shape[0]} bootstrap predictions")
    else:
        q25 = None
        q75 = None
        logger.warning("test_predictions not found in bootstrap_predictions.npz")

    logger.info(f"Test samples: {len(test_mean):,}")
    logger.info(f"Test deposits: {y_test.sum()}")

    conformal_dir = (
        PROJECT_ROOT / 'outputs' / 'conformal_oof' /
        method / f'outer_fold_{outer_fold}'
    )

    median_params_path = conformal_dir / "median_params.json"
    zone_thresholds_path = conformal_dir / "zone_thresholds.json"

    if not median_params_path.exists():
        logger.error(f"OOF median params not found: {median_params_path}")
        logger.error("Run cross_conformal_oof script first!")
        sys.exit(1)

    median_params = load_json(median_params_path)
    logger.info(f"Loaded OOF params (method: {median_params.get('method', 'unknown')}):")
    logger.info(f"  Platt coef: {median_params['platt_coef']:.4f}")
    logger.info(f"  Platt intercept: {median_params['platt_intercept']:.4f}")
    logger.info(f"  Threshold: {median_params['threshold']:.6f}")
    logger.info(f"  N positives calibration: {median_params.get('n_positives_calibration', 'N/A')}")
    logger.info(f"  Hoeffding correction: {median_params.get('hoeffding_correction', 'N/A')}")

    if zone_thresholds_path.exists():
        zone_thresholds = load_json(zone_thresholds_path)
        logger.info(f"Loaded zone thresholds (source: {zone_thresholds.get('computation_source', 'unknown')})")
    else:
        logger.warning("Zone thresholds not found, using config defaults")
        zone_thresholds = config['zone_thresholds']

    logger.info("\nApplying Platt calibration...")
    platt = PlattApplicator(
        coef=median_params['platt_coef'],
        intercept=median_params['platt_intercept']
    )
    calibrated_prob = platt.transform(test_mean)

    logger.info(f"Calibrated prob stats: mean={calibrated_prob.mean():.4f}, "
                f"std={calibrated_prob.std():.4f}")

    logger.info("\nComputing prediction sets...")
    threshold = median_params['threshold']
    prediction_sets = compute_prediction_sets(calibrated_prob, threshold)
    in_set_0 = prediction_sets[:, 0]
    in_set_1 = prediction_sets[:, 1]
    set_width = in_set_0.astype(int) + in_set_1.astype(int)

    logger.info(f"In set 1: {in_set_1.sum():,} ({in_set_1.mean() * 100:.1f}%)")
    logger.info(f"In set 0: {in_set_0.sum():,} ({in_set_0.mean() * 100:.1f}%)")

    logger.info("\nComputing zone classifications...")

    zone_method = config.get('zone_method', 'percentile_std')

    if zone_method == 'jorc30' and q25 is not None and q75 is not None:
        jorc_config = config.get('zone_thresholds', {}).get('jorc30', {})
        temp_df = pd.DataFrame({
            'bootstrap_mean': test_mean,
            'q25': q25,
            'q75': q75,
            'in_set_1': in_set_1.astype(int)
        })
        zones = assign_zones_jorc30(
            df=temp_df,
            prob_col='bootstrap_mean',
            q25_col='q25',
            q75_col='q75',
            in_set_1_col='in_set_1',
            rel_iqr_threshold=jorc_config.get('rel_iqr_threshold', 0.30),
            prob_z0=jorc_config.get('prob_z0', 0.50),
            prob_z1=jorc_config.get('prob_z1', 0.25),
            prob_z2=jorc_config.get('prob_z2', 0.10)
        )
        logger.info(f"Using JORC 30% zone classification")
    else:
        zones = compute_fnr_practical_zones(
            calibrated_prob=calibrated_prob,
            bootstrap_std=test_std,
            in_set_0=in_set_0,
            in_set_1=in_set_1,
            zone_thresholds=zone_thresholds
        )
        logger.info("Using percentile + std zone classification")

    n_zones = len(config['zone_definitions'])
    for z in range(n_zones):
        count = (zones == z).sum()
        pct = count / len(zones) * 100
        zone_name = config['zone_definitions'][z]['name']
        logger.info(f"  Zone {z} ({zone_name}): {count:,} samples ({pct:.1f}%)")

    logger.info("\nComputing metrics...")

    coverage_metrics = compute_coverage_metrics(y_test, in_set_1)
    logger.info(f"Class 1 coverage: {coverage_metrics['class_1_coverage']:.4f}")
    logger.info(f"FNR: {coverage_metrics['fnr']:.4f}")

    pr_auc = compute_pr_auc(y_test, calibrated_prob)
    logger.info(f"PR-AUC: {pr_auc:.4f}")

    zone_capture = compute_deposit_capture_by_zone(zones, y_test, n_zones=n_zones)

    logger.info("\nDeposit capture by zone:")
    for z in range(n_zones):
        stats = zone_capture[f'zone_{z}']
        logger.info(f"  Zone {z}: {stats['n_deposits']} deposits "
                    f"({stats['deposit_pct']:.1f}%), "
                    f"cumulative: {stats['cumulative_deposit_pct']:.1f}%")

    original_df = load_original_data(config)
    outer_folds = load_outer_folds(config)

    test_df = original_df.loc[test_indices]
    X_coords = test_df[config['data']['coordinate_columns'][0]].values
    Y_coords = test_df[config['data']['coordinate_columns'][1]].values

    output_dir = (
        PROJECT_ROOT / 'outputs' / 'test_evaluation_oof' /
        method / f'outer_fold_{outer_fold}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_dict = {
        'index': test_indices,
        'X': X_coords,
        'Y': Y_coords,
        'target': y_test,
        'bootstrap_mean': test_mean,
        'bootstrap_std': test_std,
        'calibrated_prob': calibrated_prob,
        'in_set_0': in_set_0.astype(int),
        'in_set_1': in_set_1.astype(int),
        'set_width': set_width,
        'zone': zones
    }

    if q25 is not None and q75 is not None:
        predictions_dict['q25'] = q25
        predictions_dict['q75'] = q75
        predictions_dict['iqr'] = q75 - q25
        predictions_dict['rel_iqr'] = (q75 - q25) / (test_mean + 1e-10)

    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    zone_df = pd.DataFrame({
        'index': test_indices,
        'X': X_coords,
        'Y': Y_coords,
        'zone': zones,
        'zone_name': [config['zone_definitions'][z]['name'] for z in zones]
    })
    zone_df.to_csv(output_dir / "zone_assignments.csv", index=False)

    metrics = {
        'outer_fold': outer_fold,
        'method': method,
        'conformal_method': 'oof_pooling',
        'n_test': len(test_mean),
        'n_deposits': int(y_test.sum()),
        'coverage': {
            'class_1_coverage': coverage_metrics['class_1_coverage'],
            'fnr': coverage_metrics['fnr']
        },
        'pr_auc': pr_auc,
        'calibration': {
            'platt_coef': median_params['platt_coef'],
            'platt_intercept': median_params['platt_intercept'],
            'threshold': threshold,
            'n_positives_calibration': median_params.get('n_positives_calibration', None),
            'hoeffding_correction': median_params.get('hoeffding_correction', None)
        },
        'zone_thresholds_source': zone_thresholds.get('computation_source', 'config'),
        'zone_method': zone_method,
        'zone_distribution': {
            **{
                f'zone_{z}': {
                    'n_samples': int((zones == z).sum()),
                    'pct': float((zones == z).mean() * 100)
                } for z in range(n_zones)
            },
            'n_zones': n_zones
        },
        'zone_capture': zone_capture,
        'prediction_sets': {
            'in_set_1_pct': float(in_set_1.mean() * 100),
            'in_set_0_pct': float(in_set_0.mean() * 100),
            'set_width_0_pct': float((set_width == 0).mean() * 100),
            'set_width_1_pct': float((set_width == 1).mean() * 100),
            'set_width_2_pct': float((set_width == 2).mean() * 100)
        }
    }
    save_json(metrics, output_dir / "metrics.json")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("[OOF] Test fold evaluation complete!")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate test fold using OOF conformal parameters"
    )
    parser.add_argument(
        "--outer-fold", type=int, default=None,
        help="Outer fold index (0-4)"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=['xgboost', 'baggingpu_xgboost'],
        help="Model method"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run for all 5 outer folds"
    )

    args = parser.parse_args()

    if args.all or args.outer_fold is None:
        for fold in range(5):
            evaluate_test_fold_oof(fold, args.method)
    else:
        evaluate_test_fold_oof(args.outer_fold, args.method)


if __name__ == "__main__":
    main()
