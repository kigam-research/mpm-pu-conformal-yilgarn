#!/usr/bin/env python3
"""02: Create Inner 5-Fold Spatial Block CV"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_spatial_blocks_for_subset(
    coords: np.ndarray,
    block_size_km: float = 50
) -> np.ndarray:
    """Create NEW spatial block IDs for a subset of data."""
    block_size_m = block_size_km * 1000

    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_norm = coords[:, 0] - x_min
    y_norm = coords[:, 1] - y_min

    block_x = (x_norm // block_size_m).astype(int)
    block_y = (y_norm // block_size_m).astype(int)

    n_blocks_x = block_x.max() + 1
    inner_block_ids = block_y * n_blocks_x + block_x

    logger.info(f"  Created {len(np.unique(inner_block_ids))} inner spatial blocks")
    logger.info(f"  Block grid: {n_blocks_x} x {block_y.max() + 1}")

    return inner_block_ids


def assign_inner_blocks_to_folds_stratified(
    inner_block_ids: np.ndarray,
    targets: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """Assign inner spatial blocks to folds with stratified deposit distribution."""
    np.random.seed(random_state)

    unique_blocks = np.unique(inner_block_ids)

    block_stats = {}
    for block_id in unique_blocks:
        mask = inner_block_ids == block_id
        block_stats[block_id] = {
            'n_samples': mask.sum(),
            'n_deposits': targets[mask].sum(),
            'indices': np.where(mask)[0]
        }

    blocks_with_deposits = [b for b in unique_blocks if block_stats[b]['n_deposits'] > 0]
    blocks_without_deposits = [b for b in unique_blocks if block_stats[b]['n_deposits'] == 0]

    logger.info(f"  Inner blocks with deposits: {len(blocks_with_deposits)}")
    logger.info(f"  Inner blocks without deposits: {len(blocks_without_deposits)}")

    inner_fold_assignments = np.full(len(inner_block_ids), -1, dtype=int)
    fold_deposit_counts = [0] * n_folds
    fold_sample_counts = [0] * n_folds

    np.random.shuffle(blocks_with_deposits)
    for block_id in blocks_with_deposits:
        stats = block_stats[block_id]
        min_fold = np.argmin(fold_deposit_counts)
        fold_deposit_counts[min_fold] += stats['n_deposits']
        fold_sample_counts[min_fold] += stats['n_samples']
        inner_fold_assignments[stats['indices']] = min_fold

    np.random.shuffle(blocks_without_deposits)
    for block_id in blocks_without_deposits:
        stats = block_stats[block_id]
        min_fold = np.argmin(fold_sample_counts)
        fold_sample_counts[min_fold] += stats['n_samples']
        inner_fold_assignments[stats['indices']] = min_fold

    return inner_fold_assignments


def create_inner_folds_for_outer_fold(
    outer_df: pd.DataFrame,
    outer_test_fold: int,
    config: dict
) -> pd.DataFrame:
    """Create inner 5-fold assignments for a specific outer test fold."""
    logger.info(f"\nCreating inner folds for outer_test_fold={outer_test_fold}")
    logger.info("-" * 50)

    train_mask = outer_df['fold_id'] != outer_test_fold
    train_df = outer_df[train_mask].copy().reset_index(drop=True)

    logger.info(f"Train data: {len(train_df):,} samples from {4} outer folds")
    logger.info(f"Deposits in train: {train_df['target'].sum()}")

    coords = train_df[['X', 'Y']].values
    targets = train_df['target'].values

    block_size_km = config['spatial_cv']['block_size_km']
    inner_block_ids = create_spatial_blocks_for_subset(coords, block_size_km)

    n_inner_folds = config['spatial_cv']['n_inner_folds']
    random_state = config['spatial_cv']['random_state'] + outer_test_fold
    inner_fold_ids = assign_inner_blocks_to_folds_stratified(
        inner_block_ids, targets, n_inner_folds, random_state
    )

    inner_df = pd.DataFrame({
        'original_index': outer_df.loc[train_mask, 'index'].values,
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'outer_fold_id': outer_df.loc[train_mask, 'fold_id'].values,
        'inner_block_id': inner_block_ids,
        'inner_fold_id': inner_fold_ids,
        'target': targets
    })

    logger.info("\n  Inner Fold Statistics:")
    for fold in range(n_inner_folds):
        fold_data = inner_df[inner_df['inner_fold_id'] == fold]
        n_samples = len(fold_data)
        n_deposits = fold_data['target'].sum()
        n_blocks = fold_data['inner_block_id'].nunique()
        pct_deposits = n_deposits / targets.sum() * 100
        logger.info(
            f"  Inner Fold {fold}: {n_samples:,} samples, "
            f"{n_deposits} deposits ({pct_deposits:.1f}%), {n_blocks} blocks"
        )

    return inner_df


def main(outer_fold: int = None):
    """Main function to create inner fold assignments."""
    logger.info("=" * 70)
    logger.info("Creating Inner 5-Fold Spatial Block CV")
    logger.info("=" * 70)

    config = load_config()

    outer_path = PROJECT_ROOT / config['output_dirs']['folds'] / "outer_fold_assignments.csv"

    if not outer_path.exists():
        logger.error(f"Outer fold assignments not found: {outer_path}")
        logger.error("Run create_outer_folds.py first!")
        sys.exit(1)

    outer_df = pd.read_csv(outer_path)
    logger.info(f"Loaded outer fold assignments: {len(outer_df):,} samples")

    n_outer_folds = config['spatial_cv']['n_outer_folds']
    if outer_fold is not None:
        outer_folds_to_process = [outer_fold]
    else:
        outer_folds_to_process = list(range(n_outer_folds))

    logger.info(f"Processing outer folds: {outer_folds_to_process}")

    for test_fold in outer_folds_to_process:
        inner_df = create_inner_folds_for_outer_fold(outer_df, test_fold, config)

        output_dir = (
            PROJECT_ROOT / config['output_dirs']['folds'] /
            "inner_folds" / f"outer_fold_{test_fold}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "inner_fold_assignments.csv"

        inner_df.to_csv(output_path, index=False)
        logger.info(f"Saved inner fold assignments to: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Inner fold creation completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create inner 5-fold spatial block CV assignments"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        default=None,
        help="Specific outer fold to process (default: all)"
    )

    args = parser.parse_args()
    main(args.outer_fold)
