#!/usr/bin/env python3
"""Zone Recomputation for OOF Conformal Results"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import load_config, save_json
from scripts.utils.conformal_utils import (
    assign_zones_jorc30,
    assign_zones_lift,
    compute_deposit_capture_by_zone
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ZONE_DEFINITIONS = {
    0: {'name': 'IMMEDIATE', 'description': 'High prob + low uncertainty'},
    1: {'name': 'PRIORITY', 'description': 'High/mid prob + uncertainty mix'},
    2: {'name': 'FOLLOW_UP', 'description': 'Mid/low prob + uncertainty mix'},
    3: {'name': 'POTENTIAL', 'description': 'Remaining conformal included'},
    4: {'name': 'EXCLUDED', 'description': 'Conformal excluded'}
}

N_ZONES = 5


def compute_zones_for_mode(df, zone_mode, config):
    """Compute zones for a given mode."""
    if zone_mode == "prob":
        jorc_config = config.get('zone_thresholds', {}).get('jorc30', {})
        return assign_zones_jorc30(
            df=df,
            prob_col='bootstrap_mean',
            q25_col='q25',
            q75_col='q75',
            in_set_1_col='in_set_1',
            rel_iqr_threshold=jorc_config.get('rel_iqr_threshold', 0.30),
            prob_z0=jorc_config.get('prob_z0', 0.50),
            prob_z1=jorc_config.get('prob_z1', 0.25),
            prob_z2=jorc_config.get('prob_z2', 0.10)
        )
    elif zone_mode == "lift":
        lift_config = config.get('zone_thresholds', {}).get('lift', {})
        return assign_zones_lift(
            df=df,
            prob_col='bootstrap_mean',
            q25_col='q25',
            q75_col='q75',
            in_set_1_col='in_set_1',
            rel_iqr_threshold=lift_config.get('rel_iqr_threshold', 0.30),
            lift_z0=lift_config.get('lift_z0', 0.01),
            lift_z1=lift_config.get('lift_z1', 0.05),
            lift_z2=lift_config.get('lift_z2', 0.10)
        )
    else:
        raise ValueError(f"Unknown zone_mode: {zone_mode}")


def recompute_zones_for_fold(outer_fold, method, config, zone_modes):
    """Recompute zones for a single fold from OOF test predictions."""
    logger.info(f"\n{'='*60}")
    logger.info(f"[OOF] Recomputing zones for {method} - Outer Fold {outer_fold}")
    logger.info(f"{'='*60}")

    test_eval_dir = (
        PROJECT_ROOT / 'outputs' / 'test_evaluation_oof' /
        method / f'outer_fold_{outer_fold}'
    )

    predictions_path = test_eval_dir / "test_predictions.csv"
    if not predictions_path.exists():
        logger.error(f"OOF predictions not found: {predictions_path}")
        return None

    df = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(df):,} samples")

    required_cols = ['bootstrap_mean', 'q25', 'q75', 'in_set_1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return None

    y_true = df['target'].values
    results = {'outer_fold': outer_fold, 'n_samples': len(df)}

    for zone_mode in zone_modes:
        logger.info(f"\n--- Zone Mode: {zone_mode.upper()} ---")

        zones = compute_zones_for_mode(df, zone_mode, config)

        logger.info(f"Zone distribution ({zone_mode}):")
        for z in range(N_ZONES):
            count = (zones == z).sum()
            pct = count / len(df) * 100
            logger.info(f"  Zone {z} ({ZONE_DEFINITIONS[z]['name']}): {count:,} ({pct:.2f}%)")

        zone_capture = compute_deposit_capture_by_zone(zones, y_true, n_zones=N_ZONES)

        logger.info(f"Deposit capture ({zone_mode}):")
        for z in range(N_ZONES):
            stats = zone_capture[f'zone_{z}']
            lift = stats['deposit_pct'] / stats['sample_pct'] if stats['sample_pct'] > 0 else 0
            logger.info(
                f"  Zone {z}: {stats['n_deposits']} deposits ({stats['deposit_pct']:.1f}%, "
                f"cumulative: {stats['cumulative_deposit_pct']:.1f}%), lift={lift:.2f}x"
            )

        practical_zone_dir = test_eval_dir / "practical_zone" / zone_mode
        practical_zone_dir.mkdir(parents=True, exist_ok=True)

        zone_df = df[['index', 'X', 'Y']].copy()
        zone_df['zone'] = zones
        zone_df.to_csv(practical_zone_dir / "zone_assignments.csv", index=False)

        n_samples_total = len(df)
        deposit_analysis = {
            'zone_mode': zone_mode,
            'conformal_method': 'oof_pooling',
            'n_zones': N_ZONES,
            'zone_capture': zone_capture,
            'zone_analysis': {}
        }

        for z in range(N_ZONES):
            zone_key = f'zone_{z}'
            zone_mask = zones == z
            deposit_analysis['zone_analysis'][zone_key] = {
                'n_samples': int(zone_mask.sum()),
                'n_deposits': int(y_true[zone_mask].sum()),
                'sample_pct': float(zone_mask.sum() / n_samples_total * 100),
                'deposit_pct': zone_capture[zone_key]['deposit_pct'],
                'cumulative_deposit_pct': zone_capture[zone_key]['cumulative_deposit_pct']
            }

        deposit_analysis['efficiency_metrics'] = {
            'zone_0_1_capture': zone_capture['zone_1']['cumulative_deposit_pct'],
            'zone_0_1_2_capture': zone_capture['zone_2']['cumulative_deposit_pct'],
            'zone_0_to_3_capture': zone_capture['zone_3']['cumulative_deposit_pct'],
            'zone_3_potential': zone_capture['zone_3']['deposit_pct'],
            'zone_4_miss_rate': zone_capture['zone_4']['deposit_pct']
        }

        deposit_analysis['validation'] = {
            'zone_0_1_capture_min_25': {
                'value': zone_capture['zone_1']['cumulative_deposit_pct'],
                'threshold': 25.0,
                'pass': zone_capture['zone_1']['cumulative_deposit_pct'] >= 25.0
            },
            'zone_0_1_2_capture_min_40': {
                'value': zone_capture['zone_2']['cumulative_deposit_pct'],
                'threshold': 40.0,
                'pass': zone_capture['zone_2']['cumulative_deposit_pct'] >= 40.0
            },
            'zone_4_capture_max_10': {
                'value': zone_capture['zone_4']['deposit_pct'],
                'threshold': 10.0,
                'pass': zone_capture['zone_4']['deposit_pct'] < 10.0
            }
        }

        save_json(deposit_analysis, practical_zone_dir / "deposit_analysis.json")
        logger.info(f"Saved: {practical_zone_dir}")

        results[zone_mode] = {
            'zone_capture': zone_capture,
            'zones': zones
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recompute zones for OOF conformal results"
    )
    parser.add_argument("--method", type=str, default="xgboost")
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--zone-mode", type=str, default="all",
                        choices=["prob", "lift", "all"])
    args = parser.parse_args()

    config = load_config()

    if args.zone_mode == "all":
        zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])
    else:
        zone_modes = [args.zone_mode]

    if args.outer_fold is not None:
        recompute_zones_for_fold(args.outer_fold, args.method, config, zone_modes)
    else:
        results = []
        for fold in range(5):
            result = recompute_zones_for_fold(fold, args.method, config, zone_modes)
            if result:
                results.append(result)

        if results:
            logger.info("\n" + "="*70)
            logger.info("[OOF] Zone Capture Summary Across All Folds")
            logger.info("="*70)
            for zone_mode in zone_modes:
                logger.info(f"\n--- {zone_mode.upper()} Mode ---")
                for z in range(N_ZONES):
                    deposit_pcts = [r[zone_mode]['zone_capture'][f'zone_{z}']['deposit_pct']
                                   for r in results if zone_mode in r]
                    cum_pcts = [r[zone_mode]['zone_capture'][f'zone_{z}']['cumulative_deposit_pct']
                               for r in results if zone_mode in r]
                    logger.info(
                        f"Zone {z} ({ZONE_DEFINITIONS[z]['name']}): "
                        f"deposit={np.mean(deposit_pcts):.1f}% +/- {np.std(deposit_pcts):.1f}%, "
                        f"cumulative={np.mean(cum_pcts):.1f}% +/- {np.std(cum_pcts):.1f}%"
                    )


if __name__ == "__main__":
    main()
