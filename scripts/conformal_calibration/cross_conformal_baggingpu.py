#!/usr/bin/env python3
"""09: Cross-Conformal Calibration for BaggingPU(XGBoost)"""

import argparse
import gc
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

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
    PlattApplicator,
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


def generate_fold_combinations(n_folds: int = 5, n_iterations: int = 20) -> List[Tuple[int, int]]:
    """Generate (calib_fold, test_fold) combinations for cross-conformal."""
    all_combinations = [
        (c, t) for c in range(n_folds) for t in range(n_folds) if c != t
    ]
    return all_combinations[:n_iterations]


def compute_zone_thresholds_dynamic(
    calibrated_prob: np.ndarray,
    bootstrap_std: np.ndarray,
    y_true: np.ndarray,
    config: dict
) -> Dict[str, float]:
    """Compute zone thresholds using config values."""
    pos_mask = y_true == 1
    n_pos = pos_mask.sum()

    config_thresholds = config.get('zone_thresholds', {})

    thresholds = {
        'prob_percentile_high': config_thresholds.get('prob_percentile_high', 98),
        'prob_percentile_mod': config_thresholds.get('prob_percentile_mod', 90),
        'prob_percentile_low': config_thresholds.get('prob_percentile_low', 70),
        'std_low': config_thresholds.get('std_low', 0.0765),
        'std_moderate': config_thresholds.get('std_moderate', 0.1020),
        'std_high': config_thresholds.get('std_high', 0.1276),
        'use_percentile_zones': config_thresholds.get('use_percentile_zones', True),
        'computation_source': 'config_percentile_zones',
        'n_positives_used': int(n_pos)
    }

    logger.info(f"Zone thresholds from config (percentile-based, {n_pos} positives in data):")
    logger.info(f"  prob percentiles: P{thresholds['prob_percentile_high']}, "
                f"P{thresholds['prob_percentile_mod']}, P{thresholds['prob_percentile_low']}")
    logger.info(f"  std (95% CI): low={thresholds['std_low']:.4f} (<30%), "
                f"mod={thresholds['std_moderate']:.4f} (<40%), high={thresholds['std_high']:.4f} (>50%)")

    return thresholds


def run_cross_conformal(
    outer_fold: int,
    config: dict = None
):
    """Run cross-conformal calibration for a specific outer fold."""
    logger.info("=" * 70)
    logger.info(f"BaggingPU(XGBoost) Cross-Conformal Calibration - Outer Fold {outer_fold}")
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
        logger.error("Run gridsearch_baggingpu_xgboost.py first!")
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
    logger.info(f"Deposits: {y_train.sum()}")

    output_dir = (
        PROJECT_ROOT / config['output_dirs']['conformal'] /
        'baggingpu_xgboost' / f'outer_fold_{outer_fold}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_inner_folds = config['spatial_cv']['n_inner_folds']
    n_iterations = config['conformal']['n_iterations']
    fold_combinations = generate_fold_combinations(n_inner_folds, n_iterations)

    logger.info(f"\nRunning {n_iterations} cross-conformal iterations...")

    alpha = config['conformal']['alpha']
    delta = config['conformal']['delta']

    iteration_results = []
    all_platt_coefs = []
    all_platt_intercepts = []
    all_thresholds = []
    all_coverages = []

    for iteration, (calib_fold, test_fold) in enumerate(fold_combinations):
        logger.info(f"\n--- Iteration {iteration + 1}/{n_iterations} "
                    f"(calib={calib_fold}, test={test_fold}) ---")

        train_mask = (inner_fold_ids != calib_fold) & (inner_fold_ids != test_fold)
        calib_mask = inner_fold_ids == calib_fold
        test_mask = inner_fold_ids == test_fold

        X_inner_train = X_train[train_mask]
        y_inner_train = y_train[train_mask]
        X_calib = X_train[calib_mask]
        y_calib = y_train[calib_mask]
        X_test = X_train[test_mask]
        y_test = y_train[test_mask]

        logger.info(f"  Inner train: {len(X_inner_train):,} samples, {y_inner_train.sum()} deposits")
        logger.info(f"  Calib: {len(X_calib):,} samples, {y_calib.sum()} deposits")
        logger.info(f"  Test: {len(X_test):,} samples, {y_test.sum()} deposits")

        preprocessor = LeakageFreePreprocessor(config)
        X_inner_train_p = preprocessor.fit_transform(X_inner_train, feature_names)
        X_calib_p = preprocessor.transform(X_calib, feature_names)
        X_test_p = preprocessor.transform(X_test, feature_names)

        model = BaggingPUClassifier(
            base_estimator_params=base_params,
            n_estimators=bagging_params['n_estimators'],
            max_samples=bagging_params['max_samples'],
            random_state=42 + iteration
        )
        model.fit(X_inner_train_p, y_inner_train)

        calib_pred = model.predict_proba(X_calib_p)[:, 1]

        platt = PlattCalibrator()
        platt.fit(calib_pred, y_calib)
        calibrated_calib = platt.transform(calib_pred)

        logger.info(f"  Platt params: coef={platt.coef_:.4f}, intercept={platt.intercept_:.4f}")

        threshold, fnr_info = fnr_control_direct(
            calibrated_calib, y_calib,
            alpha_fnr=alpha, delta=delta
        )
        logger.info(f"  FNR threshold: {threshold:.4f} (FNR={fnr_info['fnr']:.4f})")

        test_pred = model.predict_proba(X_test_p)[:, 1]
        calibrated_test = platt.transform(test_pred)
        in_set_1 = calibrated_test >= threshold

        coverage_metrics = compute_coverage_metrics(y_test, in_set_1)
        logger.info(f"  Test coverage: {coverage_metrics['class_1_coverage']:.4f}")

        all_platt_coefs.append(platt.coef_)
        all_platt_intercepts.append(platt.intercept_)
        all_thresholds.append(threshold)
        all_coverages.append(coverage_metrics['class_1_coverage'])

        iteration_results.append({
            'iteration': iteration,
            'calib_fold': calib_fold,
            'test_fold': test_fold,
            'n_inner_train': len(X_inner_train),
            'n_calib': len(X_calib),
            'n_test': len(X_test),
            'platt_coef': platt.coef_,
            'platt_intercept': platt.intercept_,
            'threshold': threshold,
            'fnr': fnr_info['fnr'],
            'fnr_upper': fnr_info['fnr_upper'],
            'test_coverage': coverage_metrics['class_1_coverage'],
            'test_fnr': coverage_metrics['fnr']
        })

        del model, preprocessor
        gc.collect()

    median_coef = float(np.median(all_platt_coefs))
    median_intercept = float(np.median(all_platt_intercepts))
    median_threshold = float(np.median(all_thresholds))
    mean_coverage = float(np.mean(all_coverages))
    std_coverage = float(np.std(all_coverages))

    logger.info("\n" + "=" * 70)
    logger.info("Cross-Conformal Results Summary")
    logger.info("=" * 70)
    logger.info(f"Median Platt coef: {median_coef:.4f}")
    logger.info(f"Median Platt intercept: {median_intercept:.4f}")
    logger.info(f"Median threshold: {median_threshold:.4f}")
    logger.info(f"Mean coverage: {mean_coverage:.4f} ± {std_coverage:.4f}")

    results_df = pd.DataFrame(iteration_results)
    results_df.to_csv(output_dir / "iteration_results.csv", index=False)

    median_params = {
        'threshold': median_threshold,
        'platt_coef': median_coef,
        'platt_intercept': median_intercept,
        'mean_coverage': mean_coverage,
        'std_coverage': std_coverage,
        'n_iterations': n_iterations,
        'alpha': alpha,
        'delta': delta
    }
    save_json(median_params, output_dir / "median_params.json")

    logger.info("\n" + "=" * 70)
    logger.info("Computing Dynamic Zone Thresholds")
    logger.info("=" * 70)

    bootstrap_dir = (
        PROJECT_ROOT / config['output_dirs']['models'] /
        'baggingpu_xgboost' / f'outer_fold_{outer_fold}'
    )
    bootstrap_pred_path = bootstrap_dir / "bootstrap_predictions.npz"

    if not bootstrap_pred_path.exists():
        logger.warning(f"Bootstrap predictions not found: {bootstrap_pred_path}")
        logger.warning("Using fallback zone thresholds from config")
        config_thresholds = config.get('zone_thresholds', {})
        zone_thresholds = {
            'prob_percentile_high': config_thresholds.get('prob_percentile_high', 98),
            'prob_percentile_mod': config_thresholds.get('prob_percentile_mod', 90),
            'prob_percentile_low': config_thresholds.get('prob_percentile_low', 70),
            'std_low': config_thresholds.get('std_low', 0.0765),
            'std_moderate': config_thresholds.get('std_moderate', 0.1020),
            'std_high': config_thresholds.get('std_high', 0.1276),
            'use_percentile_zones': config_thresholds.get('use_percentile_zones', True),
            'computation_source': 'config_fallback',
            'n_positives_used': 0
        }
    else:
        bootstrap_data = np.load(bootstrap_pred_path)
        train_mean = bootstrap_data['train_mean']
        train_std = bootstrap_data['train_std']
        y_train_bootstrap = bootstrap_data['y_train']

        platt_applicator = PlattApplicator(median_coef, median_intercept)
        calibrated_train = platt_applicator.transform(train_mean)

        zone_thresholds = compute_zone_thresholds_dynamic(
            calibrated_train, train_std, y_train_bootstrap, config
        )

    save_json(zone_thresholds, output_dir / "zone_thresholds.json")

    diagnostics = {
        'outer_fold': outer_fold,
        'n_iterations': n_iterations,
        'coverages': all_coverages,
        'thresholds': all_thresholds,
        'platt_coefs': all_platt_coefs,
        'platt_intercepts': all_platt_intercepts,
        'coverage_stats': {
            'mean': mean_coverage,
            'std': std_coverage,
            'min': float(min(all_coverages)),
            'max': float(max(all_coverages))
        },
        'threshold_stats': {
            'mean': float(np.mean(all_thresholds)),
            'std': float(np.std(all_thresholds)),
            'min': float(min(all_thresholds)),
            'max': float(max(all_thresholds))
        }
    }
    save_json(diagnostics, output_dir / "calibration_diagnostics.json")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Cross-conformal calibration complete!")

    return median_params, zone_thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-conformal calibration for BaggingPU(XGBoost)"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index (0-4)"
    )

    args = parser.parse_args()
    run_cross_conformal(args.outer_fold)


if __name__ == "__main__":
    main()
