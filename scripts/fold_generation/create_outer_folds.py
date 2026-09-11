#!/usr/bin/env python3
"""01: Create Outer 5-Fold Spatial Block CV"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import yaml
from collections import defaultdict

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


def create_spatial_blocks(coords: np.ndarray, block_size_km: float = 50) -> np.ndarray:
    """Create spatial block IDs based on coordinate grid."""
    block_size_m = block_size_km * 1000

    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_norm = coords[:, 0] - x_min
    y_norm = coords[:, 1] - y_min

    block_x = (x_norm // block_size_m).astype(int)
    block_y = (y_norm // block_size_m).astype(int)

    n_blocks_x = block_x.max() + 1
    block_ids = block_y * n_blocks_x + block_x

    logger.info(f"Created {len(np.unique(block_ids))} spatial blocks")
    logger.info(f"Block grid: {n_blocks_x} x {block_y.max() + 1}")

    return block_ids


def assign_blocks_to_folds_stratified(
    block_ids: np.ndarray,
    targets: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """Assign spatial blocks to folds with stratified deposit distribution."""
    np.random.seed(random_state)

    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    block_stats = {}
    for block_id in unique_blocks:
        mask = block_ids == block_id
        block_stats[block_id] = {
            'n_samples': mask.sum(),
            'n_deposits': targets[mask].sum(),
            'indices': np.where(mask)[0]
        }

    blocks_with_deposits = [b for b in unique_blocks if block_stats[b]['n_deposits'] > 0]
    blocks_without_deposits = [b for b in unique_blocks if block_stats[b]['n_deposits'] == 0]

    logger.info(f"Blocks with deposits: {len(blocks_with_deposits)}")
    logger.info(f"Blocks without deposits: {len(blocks_without_deposits)}")

    fold_assignments = np.full(len(block_ids), -1, dtype=int)
    fold_deposit_counts = [0] * n_folds
    fold_sample_counts = [0] * n_folds
    block_to_fold = {}

    np.random.shuffle(blocks_with_deposits)

    for block_id in blocks_with_deposits:
        stats = block_stats[block_id]

        min_fold = np.argmin(fold_deposit_counts)

        block_to_fold[block_id] = min_fold
        fold_deposit_counts[min_fold] += stats['n_deposits']
        fold_sample_counts[min_fold] += stats['n_samples']
        fold_assignments[stats['indices']] = min_fold

    np.random.shuffle(blocks_without_deposits)

    for block_id in blocks_without_deposits:
        stats = block_stats[block_id]

        min_fold = np.argmin(fold_sample_counts)

        block_to_fold[block_id] = min_fold
        fold_sample_counts[min_fold] += stats['n_samples']
        fold_assignments[stats['indices']] = min_fold

    logger.info("\nFold Statistics:")
    logger.info("-" * 50)
    for fold in range(n_folds):
        fold_mask = fold_assignments == fold
        n_samples = fold_mask.sum()
        n_deposits = targets[fold_mask].sum()
        n_blocks = len([b for b, f in block_to_fold.items() if f == fold])
        logger.info(
            f"Fold {fold}: {n_samples:,} samples, {n_deposits} deposits, "
            f"{n_blocks} blocks"
        )

    return fold_assignments


def main():
    """Main function to create outer fold assignments."""
    logger.info("=" * 70)
    logger.info("Creating Outer 5-Fold Spatial Block CV")
    logger.info("=" * 70)

    config = load_config()

    data_path = config['paths']['original_data']
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} samples")

    coord_cols = config['data']['coordinate_columns']
    target_col = config['data']['target_column']

    coords = df[coord_cols].values
    targets = df[target_col].values

    logger.info(f"Total deposits: {targets.sum()}")

    block_size_km = config['spatial_cv']['block_size_km']
    block_ids = create_spatial_blocks(coords, block_size_km)

    n_folds = config['spatial_cv']['n_outer_folds']
    random_state = config['spatial_cv']['random_state']
    fold_ids = assign_blocks_to_folds_stratified(
        block_ids, targets, n_folds, random_state
    )

    output_df = pd.DataFrame({
        'index': df.index,
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'block_id': block_ids,
        'fold_id': fold_ids,
        'target': targets
    })

    output_dir = PROJECT_ROOT / config['output_dirs']['folds']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "outer_fold_assignments.csv"

    output_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved outer fold assignments to: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Summary Statistics")
    logger.info("=" * 70)

    for fold in range(n_folds):
        fold_data = output_df[output_df['fold_id'] == fold]
        n_samples = len(fold_data)
        n_deposits = fold_data['target'].sum()
        n_blocks = fold_data['block_id'].nunique()
        pct_samples = n_samples / len(output_df) * 100
        pct_deposits = n_deposits / targets.sum() * 100

        logger.info(
            f"Fold {fold}: {n_samples:,} samples ({pct_samples:.1f}%), "
            f"{n_deposits} deposits ({pct_deposits:.1f}%), "
            f"{n_blocks} blocks"
        )

    logger.info("\nOuter fold creation completed successfully!")

    return output_df


if __name__ == "__main__":
    main()
