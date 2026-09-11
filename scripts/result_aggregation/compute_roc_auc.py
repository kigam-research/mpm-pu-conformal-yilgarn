#!/usr/bin/env python3
"""Compute ROC-AUC for Inner CV and Outer Test"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from scripts.utils import (
    load_config,
    load_original_data,
    load_outer_folds,
    load_inner_folds,
    get_feature_matrix,
    load_json,
    save_json,
    compute_roc_auc,
    BaggingPUClassifier
)
from scripts.preprocessing.leakage_free_preprocessor import LeakageFreePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_outer_test_roc_auc(method: str, outer_fold: int, config: dict) -> float:
    """Compute ROC-AUC from outer test predictions."""
    test_pred_path = (
        PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
        method / f'outer_fold_{outer_fold}' / 'test_predictions.csv'
    )

    if not test_pred_path.exists():
        logger.warning(f"Test predictions not found: {test_pred_path}")
        return None

    df = pd.read_csv(test_pred_path)

    y_true = df['target'].values
    y_prob = df['bootstrap_mean'].values

    roc_auc = compute_roc_auc(y_true, y_prob)

    return roc_auc


def compute_inner_cv_roc_auc_xgboost(outer_fold: int, config: dict) -> float:
    """Compute ROC-AUC for Inner CV using best XGBoost hyperparameters."""
    best_params_path = (
        PROJECT_ROOT / config['output_dirs']['optuna'] /
        'xgboost' / f'outer_fold_{outer_fold}' / 'best_params.json'
    )

    if not best_params_path.exists():
        logger.warning(f"Best params not found: {best_params_path}")
        return None

    best_params_data = load_json(best_params_path)
    best_params = best_params_data['best_params']

    original_df = load_original_data(config)
    outer_folds = load_outer_folds(config)
    inner_folds = load_inner_folds(outer_fold, config)

    train_indices = outer_folds[outer_folds['fold_id'] != outer_fold]['index'].values
    train_df = original_df.loc[train_indices].reset_index(drop=True)

    inner_fold_df = inner_folds.set_index('original_index')
    inner_fold_ids = inner_fold_df.loc[train_indices, 'inner_fold_id'].values

    X_train, feature_names = get_feature_matrix(train_df, config)
    y_train = train_df[config['data']['target_column']].values

    cv_scores = []
    unique_folds = np.unique(inner_fold_ids)

    for val_fold in unique_folds:
        train_mask = inner_fold_ids != val_fold
        val_mask = inner_fold_ids == val_fold

        X_inner_train = X_train[train_mask]
        y_inner_train = y_train[train_mask]
        X_inner_val = X_train[val_mask]
        y_inner_val = y_train[val_mask]

        preprocessor = LeakageFreePreprocessor(config)
        X_inner_train_p = preprocessor.fit_transform(X_inner_train, feature_names)
        X_inner_val_p = preprocessor.transform(X_inner_val, feature_names)

        n_pos = y_inner_train.sum()
        n_neg = len(y_inner_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight)
        model.fit(X_inner_train_p, y_inner_train)

        y_prob = model.predict_proba(X_inner_val_p)[:, 1]

        roc_auc = compute_roc_auc(y_inner_val, y_prob)
        cv_scores.append(roc_auc)

    return np.mean(cv_scores)


def compute_inner_cv_roc_auc_baggingpu(outer_fold: int, config: dict) -> float:
    """Compute ROC-AUC for Inner CV using best BaggingPU hyperparameters."""
    best_params_path = (
        PROJECT_ROOT / 'outputs' / 'gridsearch' /
        'baggingpu_xgboost' / f'outer_fold_{outer_fold}' / 'best_params.json'
    )

    if not best_params_path.exists():
        logger.warning(f"Best params not found: {best_params_path}")
        return None

    best_params_data = load_json(best_params_path)
    structured_params = best_params_data['best_params_structured']

    bagging_params = structured_params['bagging']
    base_params = structured_params['base_estimator']

    original_df = load_original_data(config)
    outer_folds = load_outer_folds(config)
    inner_folds = load_inner_folds(outer_fold, config)

    train_indices = outer_folds[outer_folds['fold_id'] != outer_fold]['index'].values
    train_df = original_df.loc[train_indices].reset_index(drop=True)

    inner_fold_df = inner_folds.set_index('original_index')
    inner_fold_ids = inner_fold_df.loc[train_indices, 'inner_fold_id'].values

    X_train, feature_names = get_feature_matrix(train_df, config)
    y_train = train_df[config['data']['target_column']].values

    cv_scores = []
    unique_folds = np.unique(inner_fold_ids)

    for val_fold in unique_folds:
        train_mask = inner_fold_ids != val_fold
        val_mask = inner_fold_ids == val_fold

        X_inner_train = X_train[train_mask]
        y_inner_train = y_train[train_mask]
        X_inner_val = X_train[val_mask]
        y_inner_val = y_train[val_mask]

        preprocessor = LeakageFreePreprocessor(config)
        X_inner_train_p = preprocessor.fit_transform(X_inner_train, feature_names)
        X_inner_val_p = preprocessor.transform(X_inner_val, feature_names)

        model = BaggingPUClassifier(
            base_estimator_params=base_params,
            n_estimators=bagging_params['n_estimators'],
            max_samples=bagging_params['max_samples'],
            random_state=42 + val_fold
        )
        model.fit(X_inner_train_p, y_inner_train)

        y_prob = model.predict_proba(X_inner_val_p)[:, 1]

        roc_auc = compute_roc_auc(y_inner_val, y_prob)
        cv_scores.append(roc_auc)

    return np.mean(cv_scores)


def main():
    """Main function to compute ROC-AUC for all methods and folds."""
    logger.info("=" * 70)
    logger.info("Computing ROC-AUC (Inner CV + Outer Test)")
    logger.info("=" * 70)

    config = load_config()
    n_outer_folds = config['spatial_cv']['n_outer_folds']

    methods = ['xgboost', 'baggingpu_xgboost']
    results = {}

    for method in methods:
        logger.info(f"\n{'='*70}")
        logger.info(f"Method: {method.upper()}")
        logger.info("=" * 70)

        results[method] = {
            'inner_cv': {},
            'outer_test': {}
        }

        for k in range(n_outer_folds):
            logger.info(f"\nOuter Fold {k}:")

            outer_test_auc = compute_outer_test_roc_auc(method, k, config)
            if outer_test_auc is not None:
                logger.info(f"  Outer Test ROC-AUC: {outer_test_auc:.4f}")
                results[method]['outer_test'][f'fold_{k}'] = float(outer_test_auc)
            else:
                logger.warning(f"  Outer Test ROC-AUC: NOT AVAILABLE")

            if method == 'xgboost':
                inner_cv_auc = compute_inner_cv_roc_auc_xgboost(k, config)
            elif method == 'baggingpu_xgboost':
                inner_cv_auc = compute_inner_cv_roc_auc_baggingpu(k, config)
            else:
                inner_cv_auc = None

            if inner_cv_auc is not None:
                logger.info(f"  Inner CV ROC-AUC:   {inner_cv_auc:.4f}")
                results[method]['inner_cv'][f'fold_{k}'] = float(inner_cv_auc)
            else:
                logger.warning(f"  Inner CV ROC-AUC:   NOT AVAILABLE")

        inner_cv_values = [v for v in results[method]['inner_cv'].values() if v is not None]
        outer_test_values = [v for v in results[method]['outer_test'].values() if v is not None]

        if inner_cv_values:
            results[method]['inner_cv_mean'] = float(np.mean(inner_cv_values))
            results[method]['inner_cv_std'] = float(np.std(inner_cv_values))
        if outer_test_values:
            results[method]['outer_test_mean'] = float(np.mean(outer_test_values))
            results[method]['outer_test_std'] = float(np.std(outer_test_values))

    output_dir = PROJECT_ROOT / config['output_dirs']['aggregated']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'roc_auc_results.json'
    save_json(results, output_path)
    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE: Inner CV vs Outer Test ROC-AUC")
    logger.info("=" * 70)

    for method in methods:
        logger.info(f"\n{method.upper()}:")
        logger.info("-" * 70)
        logger.info(f"{'Fold':<15} {'Inner CV':<15} {'Outer Test':<15} {'Difference':<15}")
        logger.info("-" * 70)

        for k in range(n_outer_folds):
            inner_val = results[method]['inner_cv'].get(f'fold_{k}')
            outer_val = results[method]['outer_test'].get(f'fold_{k}')

            inner_str = f"{inner_val:.4f}" if inner_val is not None else "N/A"
            outer_str = f"{outer_val:.4f}" if outer_val is not None else "N/A"

            if inner_val is not None and outer_val is not None:
                diff = outer_val - inner_val
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "N/A"

            logger.info(f"Fold {k:<9} {inner_str:<15} {outer_str:<15} {diff_str:<15}")

        logger.info("-" * 70)

        inner_mean = results[method].get('inner_cv_mean')
        inner_std = results[method].get('inner_cv_std')
        outer_mean = results[method].get('outer_test_mean')
        outer_std = results[method].get('outer_test_std')

        if inner_mean is not None and outer_mean is not None:
            inner_summary = f"{inner_mean:.4f} ± {inner_std:.4f}"
            outer_summary = f"{outer_mean:.4f} ± {outer_std:.4f}"
            diff_mean = outer_mean - inner_mean
            diff_str = f"{diff_mean:+.4f}"
        else:
            inner_summary = "N/A"
            outer_summary = "N/A"
            diff_str = "N/A"

        logger.info(f"{'Mean ± Std':<15} {inner_summary:<15} {outer_summary:<15} {diff_str:<15}")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
