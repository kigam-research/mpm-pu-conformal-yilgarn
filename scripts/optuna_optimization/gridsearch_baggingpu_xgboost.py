#!/usr/bin/env python3
"""Grid Search Hyperparameter Optimization for BaggingPU(XGBoost)"""

import argparse
import gc
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import (
    load_config,
    load_hyperparameter_ranges,
    load_original_data,
    load_outer_folds,
    load_inner_folds,
    get_feature_matrix,
    load_json,
    save_json,
    compute_pr_auc,
    BaggingPUClassifier
)
from scripts.preprocessing.leakage_free_preprocessor import LeakageFreePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_grid_search_params(hp_ranges: dict) -> tuple:
    """Get grid search parameters from hyperparameter_ranges.yaml."""
    gridsearch_config = hp_ranges.get('baggingpu_gridsearch', {})

    max_samples_config = gridsearch_config.get('max_samples', {})
    grid_max_samples = max_samples_config.get(
        'values',
        [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    )

    n_estimators_config = gridsearch_config.get('n_estimators', {})
    grid_n_estimators = n_estimators_config.get(
        'values',
        [10, 20, 30, 40, 50]
    )

    return grid_max_samples, grid_n_estimators


def evaluate_combination(
    bagging_n_estimators: int,
    bagging_max_samples: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_fold_ids: np.ndarray,
    feature_names: list,
    config: dict,
    base_params: dict
) -> float:
    """Evaluate a single parameter combination using inner 5-fold CV."""
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
            n_estimators=bagging_n_estimators,
            max_samples=bagging_max_samples,
            random_state=42 + val_fold
        )
        model.fit(X_inner_train_p, y_inner_train)

        y_prob = model.predict_proba(X_inner_val_p)[:, 1]

        pr_auc = compute_pr_auc(y_inner_val, y_prob)
        cv_scores.append(pr_auc)

    gc.collect()

    return np.mean(cv_scores)


def run_gridsearch(
    outer_fold: int,
    config: dict = None
):
    """Run grid search for a specific outer fold."""
    logger.info("=" * 70)
    logger.info(f"BaggingPU(XGBoost) Grid Search - Outer Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    hp_ranges = load_hyperparameter_ranges()

    xgb_optuna_dir = (
        PROJECT_ROOT / config['output_dirs']['optuna'] /
        'xgboost' / f'outer_fold_{outer_fold}'
    )
    xgb_best_params_path = xgb_optuna_dir / "best_params.json"

    if not xgb_best_params_path.exists():
        logger.error(f"XGBoost best params not found: {xgb_best_params_path}")
        logger.error("Run optuna_xgboost.py first to optimize XGBoost parameters!")
        sys.exit(1)

    xgb_best_data = load_json(xgb_best_params_path)
    xgb_best_params = xgb_best_data['best_params']
    logger.info(f"Loaded XGBoost best params (PR-AUC: {xgb_best_data['best_value']:.4f})")
    logger.info(f"  XGBoost: n_estimators={xgb_best_params['n_estimators']}, "
                f"max_depth={xgb_best_params['max_depth']}, "
                f"learning_rate={xgb_best_params['learning_rate']:.4f}")

    xgb_fixed = hp_ranges['xgboost'].get('fixed', {})
    base_params = {**xgb_best_params, **xgb_fixed}

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
    logger.info(f"Inner folds: {np.unique(inner_fold_ids)}")

    output_dir = (
        PROJECT_ROOT / 'outputs' / 'gridsearch' /
        'baggingpu_xgboost' / f'outer_fold_{outer_fold}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_max_samples, grid_n_estimators = get_grid_search_params(hp_ranges)

    n_combinations = len(grid_max_samples) * len(grid_n_estimators)
    logger.info(f"\nStarting grid search with {n_combinations} combinations...")
    logger.info(f"  max_samples: {grid_max_samples}")
    logger.info(f"  n_estimators: {grid_n_estimators}")

    results = []
    best_score = -np.inf
    best_params = None
    best_idx = -1

    combination_idx = 0
    for max_samples in grid_max_samples:
        for n_estimators in grid_n_estimators:
            combination_idx += 1

            logger.info(f"\n[{combination_idx}/{n_combinations}] Evaluating: "
                       f"n_estimators={n_estimators}, max_samples={max_samples}")

            mean_pr_auc = evaluate_combination(
                bagging_n_estimators=n_estimators,
                bagging_max_samples=max_samples,
                X_train=X_train,
                y_train=y_train,
                inner_fold_ids=inner_fold_ids,
                feature_names=feature_names,
                config=config,
                base_params=base_params
            )

            logger.info(f"  Mean PR-AUC: {mean_pr_auc:.4f}")

            results.append({
                'combination_idx': combination_idx,
                'bagging_n_estimators': n_estimators,
                'bagging_max_samples': max_samples,
                'mean_pr_auc': mean_pr_auc
            })

            if mean_pr_auc > best_score:
                best_score = mean_pr_auc
                best_params = {
                    'bagging_n_estimators': n_estimators,
                    'bagging_max_samples': max_samples
                }
                best_idx = combination_idx

    logger.info("\n" + "=" * 70)
    logger.info("Grid Search Complete")
    logger.info("=" * 70)
    logger.info(f"Best combination: {best_idx}/{n_combinations}")
    logger.info(f"Best PR-AUC: {best_score:.4f}")
    logger.info(f"Best params: {best_params}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "gridsearch_results.csv", index=False)
    logger.info(f"\nGrid search results saved to: {output_dir / 'gridsearch_results.csv'}")

    structured_params = {
        'bagging': {
            'n_estimators': best_params['bagging_n_estimators'],
            'max_samples': best_params['bagging_max_samples']
        },
        'base_estimator': base_params
    }

    save_data = {
        'best_trial': best_idx,
        'best_value': best_score,
        'best_params_flat': best_params,
        'best_params_structured': structured_params,
        'outer_fold': outer_fold,
        'n_combinations': n_combinations
    }
    save_json(save_data, output_dir / "best_params.json")
    logger.info(f"Best params saved to: {output_dir / 'best_params.json'}")

    return structured_params, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search for BaggingPU(XGBoost)"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index (0-4)"
    )

    args = parser.parse_args()

    config = load_config()
    run_gridsearch(args.outer_fold, config)


if __name__ == "__main__":
    main()
