#!/usr/bin/env python3
"""06: Train XGBoost Bootstrap Ensemble"""

import argparse
import gc
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from scripts.utils import (
    load_config,
    load_hyperparameter_ranges,
    load_original_data,
    load_outer_folds,
    get_feature_matrix,
    load_json,
    save_numpy,
    save_json
)
from scripts.preprocessing.leakage_free_preprocessor import LeakageFreePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_bootstrap_ensemble(
    outer_fold: int,
    config: dict = None
):
    """Train bootstrap ensemble for a specific outer fold."""
    logger.info("=" * 70)
    logger.info(f"XGBoost Bootstrap Ensemble - Outer Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    hp_ranges = load_hyperparameter_ranges()
    xgb_fixed = hp_ranges['xgboost'].get('fixed', {})

    optuna_dir = (
        PROJECT_ROOT / config['output_dirs']['optuna'] /
        'xgboost' / f'outer_fold_{outer_fold}'
    )
    best_params_path = optuna_dir / "best_params.json"

    if not best_params_path.exists():
        logger.error(f"Best params not found: {best_params_path}")
        logger.error("Run optuna_xgboost.py first!")
        sys.exit(1)

    best_params_data = load_json(best_params_path)
    best_params = best_params_data['best_params']

    best_params.update(xgb_fixed)
    best_params['n_jobs'] = -1

    logger.info(f"Loaded best params from Optuna (PR-AUC: {best_params_data['best_value']:.4f})")
    logger.info(f"Fixed params from config: {list(xgb_fixed.keys())}")

    original_df = load_original_data(config)
    outer_folds = load_outer_folds(config)

    train_mask = outer_folds['fold_id'] != outer_fold
    test_mask = outer_folds['fold_id'] == outer_fold

    train_indices = outer_folds[train_mask]['index'].values
    test_indices = outer_folds[test_mask]['index'].values

    train_df = original_df.loc[train_indices].reset_index(drop=True)
    test_df = original_df.loc[test_indices].reset_index(drop=True)

    X_train, feature_names = get_feature_matrix(train_df, config)
    X_test, _ = get_feature_matrix(test_df, config)
    y_train = train_df[config['data']['target_column']].values
    y_test = test_df[config['data']['target_column']].values

    logger.info(f"Train: {len(X_train):,} samples, {y_train.sum()} deposits")
    logger.info(f"Test: {len(X_test):,} samples, {y_test.sum()} deposits")

    logger.info("Preprocessing data...")
    preprocessor = LeakageFreePreprocessor(config)
    X_train_p = preprocessor.fit_transform(X_train, feature_names)
    X_test_p = preprocessor.transform(X_test, feature_names)

    output_dir = (
        PROJECT_ROOT / config['output_dirs']['models'] /
        'xgboost' / f'outer_fold_{outer_fold}'
    )
    model_dir = output_dir / "bootstrap_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    n_bootstraps = config['bootstrap']['n_bootstraps']
    random_state = config['bootstrap']['random_state']
    np.random.seed(random_state)

    all_train_predictions = []
    all_test_predictions = []

    logger.info(f"\nTraining {n_bootstraps} bootstrap models...")

    for i in range(n_bootstraps):
        boot_idx = np.random.choice(len(X_train_p), len(X_train_p), replace=True)
        X_boot = X_train_p[boot_idx]
        y_boot = y_train[boot_idx]

        model = XGBClassifier(**best_params)
        model.fit(X_boot, y_boot)

        model_path = model_dir / f"bootstrap_{i:03d}.joblib"
        joblib.dump(model, model_path)

        train_pred = model.predict_proba(X_train_p)[:, 1]
        all_train_predictions.append(train_pred)

        test_pred = model.predict_proba(X_test_p)[:, 1]
        all_test_predictions.append(test_pred)

        if (i + 1) % 10 == 0:
            logger.info(f"  Trained {i + 1}/{n_bootstraps} bootstrap models")
            gc.collect()

    all_train_predictions = np.array(all_train_predictions)
    all_test_predictions = np.array(all_test_predictions)

    train_mean = all_train_predictions.mean(axis=0)
    train_std = all_train_predictions.std(axis=0)
    test_mean = all_test_predictions.mean(axis=0)
    test_std = all_test_predictions.std(axis=0)

    logger.info(f"\nTest predictions - Mean: {test_mean.mean():.4f}, Std: {test_std.mean():.4f}")

    save_numpy(
        {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_predictions': all_train_predictions,
            'test_predictions': all_test_predictions,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'y_train': y_train,
            'y_test': y_test
        },
        output_dir / "bootstrap_predictions.npz"
    )

    save_json(preprocessor.get_params(), output_dir / "preprocessor_params.json")

    summary = {
        'outer_fold': outer_fold,
        'n_bootstraps': n_bootstraps,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_deposits': int(y_train.sum()),
        'test_deposits': int(y_test.sum()),
        'test_mean_pred': float(test_mean.mean()),
        'test_std_pred': float(test_std.mean()),
        'best_params': best_params
    }
    save_json(summary, output_dir / "training_summary.json")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Bootstrap training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost bootstrap ensemble"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index (0-4)"
    )

    args = parser.parse_args()
    train_bootstrap_ensemble(args.outer_fold)


if __name__ == "__main__":
    main()
