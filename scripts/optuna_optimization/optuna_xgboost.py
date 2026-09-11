#!/usr/bin/env python3
"""04: Optuna Hyperparameter Optimization for XGBoost"""

import argparse
import gc
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from xgboost import XGBClassifier
import yaml

from scripts.utils import (
    load_config,
    load_hyperparameter_ranges,
    load_original_data,
    load_outer_folds,
    load_inner_folds,
    get_feature_matrix,
    save_json,
    compute_pr_auc
)
from scripts.preprocessing.leakage_free_preprocessor import LeakageFreePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_fold_ids: np.ndarray,
    feature_names: list,
    hp_ranges: dict,
    config: dict
):
    """Create Optuna objective function for XGBoost optimization."""
    n_inner_folds = config['spatial_cv']['n_inner_folds']
    xgb_ranges = hp_ranges['xgboost']
    xgb_fixed = hp_ranges['xgboost'].get('fixed', {})

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators',
                xgb_ranges['n_estimators']['low'],
                xgb_ranges['n_estimators']['high'],
                step=xgb_ranges['n_estimators'].get('step', 1)
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                xgb_ranges['max_depth']['low'],
                xgb_ranges['max_depth']['high']
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                xgb_ranges['learning_rate']['low'],
                xgb_ranges['learning_rate']['high'],
                log=xgb_ranges['learning_rate'].get('log', False)
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                xgb_ranges['min_child_weight']['low'],
                xgb_ranges['min_child_weight']['high']
            ),
            'gamma': trial.suggest_float(
                'gamma',
                xgb_ranges['gamma']['low'],
                xgb_ranges['gamma']['high']
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                xgb_ranges['reg_alpha']['low'],
                xgb_ranges['reg_alpha']['high'],
                log=xgb_ranges['reg_alpha'].get('log', False)
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                xgb_ranges['reg_lambda']['low'],
                xgb_ranges['reg_lambda']['high'],
                log=xgb_ranges['reg_lambda'].get('log', False)
            ),
            **xgb_fixed
        }

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

            model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
            model.fit(X_inner_train_p, y_inner_train)

            y_prob = model.predict_proba(X_inner_val_p)[:, 1]

            pr_auc = compute_pr_auc(y_inner_val, y_prob)
            cv_scores.append(pr_auc)

            trial.report(np.mean(cv_scores), len(cv_scores) - 1)

            if trial.should_prune():
                raise optuna.TrialPruned()

        gc.collect()

        return np.mean(cv_scores)

    return objective


def run_optimization(
    outer_fold: int,
    n_trials: int = 50,
    config: dict = None
):
    """Run Optuna optimization for a specific outer fold."""
    logger.info("=" * 70)
    logger.info(f"XGBoost Optuna Optimization - Outer Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    hp_ranges = load_hyperparameter_ranges()

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
        PROJECT_ROOT / config['output_dirs']['optuna'] /
        'xgboost' / f'outer_fold_{outer_fold}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    study_path = output_dir / "study.db"
    study = optuna.create_study(
        study_name=f"xgboost_outer_fold_{outer_fold}",
        direction="maximize",
        sampler=TPESampler(seed=config['random_seeds']['optuna']),
        pruner=MedianPruner(
            n_startup_trials=config['optuna']['pruner_n_startup_trials'],
            n_warmup_steps=config['optuna']['pruner_n_warmup_steps']
        ),
        storage=f"sqlite:///{study_path}",
        load_if_exists=True
    )

    objective = create_objective(
        X_train, y_train, inner_fold_ids, feature_names, hp_ranges, config
    )

    logger.info(f"\nStarting optimization with {n_trials} trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )

    logger.info("\n" + "=" * 70)
    logger.info("Optimization Complete")
    logger.info("=" * 70)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    best_params = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'outer_fold': outer_fold,
        'n_trials': n_trials
    }
    save_json(best_params, output_dir / "best_params.json")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optimization_history.csv", index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna optimization for XGBoost"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index (0-4)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of optimization trials (default: from config)"
    )

    args = parser.parse_args()

    config = load_config()
    n_trials = args.n_trials or config['optuna']['n_trials']

    run_optimization(args.outer_fold, n_trials, config)


if __name__ == "__main__":
    main()
