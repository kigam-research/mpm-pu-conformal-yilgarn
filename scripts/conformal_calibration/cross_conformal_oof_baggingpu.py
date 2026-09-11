#!/usr/bin/env python3
"""Cross-Conformal Calibration for BaggingPU(XGBoost) - OOF Pooling Method (Alternative 2)"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import (
    load_config,
    load_original_data,
    load_outer_folds,
    load_inner_folds,
    get_feature_matrix,
    load_json,
    save_json,
    PlattCalibrator,
    fnr_control_direct,
    compute_coverage_metrics,
    BaggingPUClassifier
)
from scripts.preprocessing.leakage_free_preprocessor import LeakageFreePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_oof_conformal(
    outer_fold: int,
    config: dict = None
):
    """Run OOF Pooling cross-conformal calibration for BaggingPU(XGBoost)."""
    logger.info("=" * 70)
    logger.info(f"BaggingPU(XGBoost) OOF Pooling Cross-Conformal - Outer Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    gridsearch_dir = (
        PROJECT_ROOT / 'outputs' / 'gridsearch' /
        'baggingpu_xgboost' / f'outer_fold_{outer_fold}'
    )
    best_params_path = gridsearch_dir / "best_params.json"

    if not best_params_path.exists():
        logger.error(f"Best params not found: {best_params_path}")
        sys.exit(1)

    best_params_data = load_json(best_params_path)
    structured_params = best_params_data['best_params_structured']
    bagging_params = structured_params['bagging']
    base_params = structured_params['base_estimator']

    logger.info(f"Loaded best params from Grid Search (PR-AUC: {best_params_data['best_value']:.4f})")
    logger.info(f"  Bagging: n_estimators={bagging_params['n_estimators']}, "
                f"max_samples={bagging_params['max_samples']:.3f}")

    original_df = load_original_data(config)
    outer_folds = load_outer_folds(config)
    inner_folds = load_inner_folds(outer_fold, config)

    train_indices = outer_folds[outer_folds['fold_id'] != outer_fold]['index'].values
    train_df = original_df.loc[train_indices].reset_index(drop=True)

    inner_fold_df = inner_folds.set_index('original_index')
    inner_fold_ids = inner_fold_df.loc[train_indices, 'inner_fold_id'].values

    X_train, feature_names = get_feature_matrix(train_df, config)
    y_train = train_df[config['data']['target_column']].values

    logger.info(f"Training data: {len(X_train):,} samples, {len(feature_names)} features")
    logger.info(f"Deposits: {int(y_train.sum())}")

    output_dir = (
        PROJECT_ROOT / 'outputs' / 'conformal_oof' /
        'baggingpu_xgboost' / f'outer_fold_{outer_fold}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_inner_folds = config['spatial_cv']['n_inner_folds']
    alpha = config['conformal']['alpha']
    delta = config['conformal']['delta']

    logger.info(f"\nCollecting OOF predictions across {n_inner_folds} inner folds...")

    oof_predictions = np.zeros(len(X_train))
    oof_fold_ids = np.zeros(len(X_train), dtype=int)
    fold_metrics = []

    for k in range(n_inner_folds):
        logger.info(f"\n--- Inner fold {k}/{n_inner_folds - 1} (predict fold {k}) ---")

        train_mask = inner_fold_ids != k
        pred_mask = inner_fold_ids == k

        X_fold_train = X_train[train_mask]
        y_fold_train = y_train[train_mask]
        X_fold_pred = X_train[pred_mask]
        y_fold_pred = y_train[pred_mask]

        logger.info(f"  Train: {len(X_fold_train):,} samples, {int(y_fold_train.sum())} deposits")
        logger.info(f"  Predict: {len(X_fold_pred):,} samples, {int(y_fold_pred.sum())} deposits")

        preprocessor = LeakageFreePreprocessor(config)
        X_fold_train_p = preprocessor.fit_transform(X_fold_train, feature_names)
        X_fold_pred_p = preprocessor.transform(X_fold_pred, feature_names)

        model = BaggingPUClassifier(
            base_estimator_params=base_params,
            n_estimators=bagging_params['n_estimators'],
            max_samples=bagging_params['max_samples'],
            random_state=42
        )
        model.fit(X_fold_train_p, y_fold_train)

        fold_pred = model.predict_proba(X_fold_pred_p)[:, 1]
        oof_predictions[pred_mask] = fold_pred
        oof_fold_ids[pred_mask] = k

        logger.info(f"  Predictions: mean={fold_pred.mean():.6f}, "
                    f"std={fold_pred.std():.6f}, "
                    f"pos_mean={fold_pred[y_fold_pred == 1].mean():.6f}")

        fold_metrics.append({
            'inner_fold': k,
            'n_train': len(X_fold_train),
            'n_pred': len(X_fold_pred),
            'n_deposits_pred': int(y_fold_pred.sum()),
            'pred_mean': float(fold_pred.mean()),
            'pred_std': float(fold_pred.std()),
            'pos_pred_mean': float(fold_pred[y_fold_pred == 1].mean())
        })

        del model, preprocessor
        gc.collect()

    logger.info("\n" + "=" * 70)
    logger.info("Fitting Platt calibration on pooled OOF predictions")
    logger.info("=" * 70)

    platt = PlattCalibrator()
    platt.fit(oof_predictions, y_train)
    calibrated_oof = platt.transform(oof_predictions)

    platt_coef = platt.coef_
    platt_intercept = platt.intercept_

    logger.info(f"Platt params: coef={platt_coef:.4f}, intercept={platt_intercept:.4f}")

    logger.info("\nComputing FNR threshold on pooled calibrated OOF...")

    threshold, fnr_info = fnr_control_direct(
        calibrated_oof, y_train,
        alpha_fnr=alpha, delta=delta
    )

    logger.info(f"FNR threshold: {threshold:.6f}")
    logger.info(f"FNR: {fnr_info['fnr']:.4f}, FNR upper: {fnr_info['fnr_upper']:.4f}")
    logger.info(f"Hoeffding: {fnr_info['hoeffding']:.4f}, N pos: {fnr_info['n_pos']}")

    in_set_1_oof = calibrated_oof >= threshold
    oof_coverage = compute_coverage_metrics(y_train, in_set_1_oof)
    logger.info(f"OOF coverage: {oof_coverage['class_1_coverage']:.4f}")

    oof_df = pd.DataFrame({
        'sample_idx': np.arange(len(X_train)),
        'original_index': train_indices,
        'inner_fold_id': oof_fold_ids,
        'y_true': y_train,
        'oof_raw_prob': oof_predictions,
        'oof_calibrated_prob': calibrated_oof,
        'in_set_1': in_set_1_oof.astype(int)
    })
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    median_params = {
        'threshold': float(threshold),
        'platt_coef': float(platt_coef),
        'platt_intercept': float(platt_intercept),
        'mean_coverage': float(oof_coverage['class_1_coverage']),
        'std_coverage': 0.0,
        'n_iterations': 1,
        'alpha': alpha,
        'delta': delta,
        'method': 'oof_pooling',
        'n_positives_calibration': int(fnr_info['n_pos']),
        'hoeffding_correction': float(fnr_info['hoeffding'])
    }
    save_json(median_params, output_dir / "median_params.json")

    config_thresholds = config.get('zone_thresholds', {})
    zone_thresholds = {
        'prob_percentile_high': config_thresholds.get('percentile_std', {}).get('prob_percentile_high', 98),
        'prob_percentile_mod': config_thresholds.get('percentile_std', {}).get('prob_percentile_mod', 90),
        'prob_percentile_low': config_thresholds.get('percentile_std', {}).get('prob_percentile_low', 70),
        'std_low': config_thresholds.get('percentile_std', {}).get('std_low', 0.0765),
        'std_moderate': config_thresholds.get('percentile_std', {}).get('std_moderate', 0.1020),
        'std_high': config_thresholds.get('percentile_std', {}).get('std_high', 0.1276),
        'use_percentile_zones': True,
        'computation_source': 'config_percentile_zones'
    }
    save_json(zone_thresholds, output_dir / "zone_thresholds.json")

    diagnostics = {
        'outer_fold': outer_fold,
        'method': 'oof_pooling',
        'model': 'baggingpu_xgboost',
        'n_inner_folds': n_inner_folds,
        'n_train_samples': len(X_train),
        'n_positives': int(y_train.sum()),
        'platt_coef': float(platt_coef),
        'platt_intercept': float(platt_intercept),
        'threshold': float(threshold),
        'fnr': float(fnr_info['fnr']),
        'fnr_upper': float(fnr_info['fnr_upper']),
        'hoeffding': float(fnr_info['hoeffding']),
        'n_pos_calibration': int(fnr_info['n_pos']),
        'oof_coverage': float(oof_coverage['class_1_coverage']),
        'fold_metrics': fold_metrics
    }
    save_json(diagnostics, output_dir / "calibration_diagnostics.json")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("OOF Pooling cross-conformal calibration complete!")

    return median_params, zone_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Run OOF Pooling cross-conformal calibration for BaggingPU(XGBoost)"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        default=None,
        help="Outer fold index (0-4), or all folds if not specified"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all 5 outer folds"
    )

    args = parser.parse_args()

    if args.all or args.outer_fold is None:
        for fold in range(5):
            run_oof_conformal(fold)
    else:
        run_oof_conformal(args.outer_fold)


if __name__ == "__main__":
    main()
