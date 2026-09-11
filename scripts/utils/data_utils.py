#!/usr/bin/env python3
"""Data Utilities"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_hyperparameter_ranges() -> dict:
    """Load hyperparameter ranges from yaml."""
    config_path = get_project_root() / "config" / "hyperparameter_ranges.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_original_data(config: Optional[dict] = None) -> pd.DataFrame:
    """Load the original Yilgarn dataset."""
    if config is None:
        config = load_config()

    data_path = config['paths']['original_data']

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    df = pd.read_csv(data_path)
    logger.info(f"Loaded original data: {len(df):,} samples")
    return df


def load_outer_folds(config: Optional[dict] = None) -> pd.DataFrame:
    """Load outer fold assignments."""
    if config is None:
        config = load_config()

    project_root = get_project_root()
    fold_path = project_root / config['output_dirs']['folds'] / "outer_fold_assignments.csv"

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Outer fold assignments not found: {fold_path}\n"
            "Run create_outer_folds.py first!"
        )

    df = pd.read_csv(fold_path)
    logger.info(f"Loaded outer fold assignments: {len(df):,} samples")
    return df


def load_inner_folds(outer_fold: int, config: Optional[dict] = None) -> pd.DataFrame:
    """Load inner fold assignments for a specific outer fold."""
    if config is None:
        config = load_config()

    project_root = get_project_root()
    fold_path = (
        project_root / config['output_dirs']['folds'] /
        "inner_folds" / f"outer_fold_{outer_fold}" / "inner_fold_assignments.csv"
    )

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Inner fold assignments not found: {fold_path}\n"
            "Run create_inner_folds.py first!"
        )

    df = pd.read_csv(fold_path)
    logger.info(f"Loaded inner fold assignments for outer_fold={outer_fold}: {len(df):,} samples")
    return df


def get_train_test_data(
    outer_fold: int,
    config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Get train and test data for a specific outer fold."""
    if config is None:
        config = load_config()

    original_df = load_original_data(config)
    outer_folds = load_outer_folds(config)

    df = original_df.merge(
        outer_folds[['index', 'fold_id']],
        left_index=True,
        right_on='index'
    )

    train_mask = df['fold_id'] != outer_fold
    test_mask = df['fold_id'] == outer_fold

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    target_col = config['data']['target_column']
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    logger.info(
        f"Outer fold {outer_fold}: "
        f"Train={len(train_df):,} (deposits={y_train.sum()}), "
        f"Test={len(test_df):,} (deposits={y_test.sum()})"
    )

    return train_df, test_df, y_train, y_test


def get_feature_matrix(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> Tuple[np.ndarray, List[str]]:
    """Extract feature matrix from DataFrame."""
    if config is None:
        config = load_config()

    feature_cols = config['data']['feature_columns']
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].values
    return X, available_cols


def save_json(data: dict, path: Union[str, Path]) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved JSON to: {path}")


def load_json(path: Union[str, Path]) -> dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_numpy(data: dict, path: Union[str, Path]) -> None:
    """Save numpy arrays to npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, **data)
    logger.info(f"Saved numpy arrays to: {path}")


def load_numpy(path: Union[str, Path]) -> dict:
    """Load numpy arrays from npz file."""
    data = np.load(path, allow_pickle=True)
    return dict(data)
