#!/usr/bin/env python3
"""Zone recomputation script (prob and lift modes)"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import load_config, load_json, save_json
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


def get_zone_definitions(config: dict) -> dict:
    """Get zone definitions from config.yaml."""
    zone_defs_config = config.get('zone_definitions', {})

    zone_definitions = {}
    for zone_id, zone_info in zone_defs_config.items():
        zone_definitions[int(zone_id)] = {
            'name': zone_info.get('name', f'ZONE_{zone_id}'),
            'description': zone_info.get('description', '')
        }

    if not zone_definitions:
        zone_definitions = {
            0: {'name': 'IMMEDIATE', 'description': 'prob >= 0.50 AND rel_IQR <= 30%'},
            1: {'name': 'PRIORITY', 'description': 'high prob OR mid prob with low uncertainty'},
            2: {'name': 'FOLLOW_UP', 'description': 'mid prob OR low-mid prob with low uncertainty'},
            3: {'name': 'POTENTIAL', 'description': 'remaining in_set_1 samples'},
            4: {'name': 'EXCLUDED', 'description': 'in_set_1 = 0 (conformal excluded)'}
        }

    return zone_definitions


def get_n_zones(config: dict) -> int:
    """Get number of zones from config.yaml."""
    zone_defs = config.get('zone_definitions', {})
    if zone_defs:
        return len(zone_defs)
    return 5


def get_validation_criteria(config: dict) -> dict:
    """Get validation criteria from config.yaml."""
    criteria = config.get('validation_criteria', {})

    return {
        'zone_0_1_capture_min': criteria.get('zone_0_1_capture_min', 0.25),
        'zone_0_1_2_capture_min': criteria.get('zone_0_1_2_capture_min', 0.40),
        'zone_4_capture_max': criteria.get('zone_4_capture_max', 0.10)
    }


def compute_zones_for_mode(
    df: pd.DataFrame,
    zone_mode: str,
    config: dict
) -> np.ndarray:
    """Compute zones for the given zone_mode."""
    if zone_mode == "prob":
        prob_config = config.get('zone_thresholds', {}).get('prob', {})
        rel_iqr_threshold = prob_config.get('rel_iqr_threshold', 0.30)
        prob_z0 = prob_config.get('prob_z0', 0.50)
        prob_z1 = prob_config.get('prob_z1', 0.25)
        prob_z2 = prob_config.get('prob_z2', 0.10)

        logger.info(f"  Prob mode thresholds:")
        logger.info(f"    rel_IQR: {rel_iqr_threshold:.0%}")
        logger.info(f"    prob_z0: {prob_z0}, prob_z1: {prob_z1}, prob_z2: {prob_z2}")

        return assign_zones_jorc30(
            df=df,
            prob_col='bootstrap_mean',
            q25_col='q25',
            q75_col='q75',
            in_set_1_col='in_set_1',
            rel_iqr_threshold=rel_iqr_threshold,
            prob_z0=prob_z0,
            prob_z1=prob_z1,
            prob_z2=prob_z2
        )

    elif zone_mode == "lift":
        lift_config = config.get('zone_thresholds', {}).get('lift', {})
        rel_iqr_threshold = lift_config.get('rel_iqr_threshold', 0.30)
        lift_z0 = lift_config.get('lift_z0', 0.01)
        lift_z1 = lift_config.get('lift_z1', 0.05)
        lift_z2 = lift_config.get('lift_z2', 0.10)

        logger.info(f"  Lift mode thresholds:")
        logger.info(f"    rel_IQR: {rel_iqr_threshold:.0%}")
        logger.info(f"    Top {lift_z0*100:.0f}%, {lift_z1*100:.0f}%, {lift_z2*100:.0f}%")

        return assign_zones_lift(
            df=df,
            prob_col='bootstrap_mean',
            q25_col='q25',
            q75_col='q75',
            in_set_1_col='in_set_1',
            rel_iqr_threshold=rel_iqr_threshold,
            lift_z0=lift_z0,
            lift_z1=lift_z1,
            lift_z2=lift_z2
        )

    else:
        raise ValueError(f"Unknown zone_mode: {zone_mode}")


def recompute_zones_for_fold(
    outer_fold: int,
    method: str,
    config: dict,
    zone_modes: List[str],
    zone_definitions: dict = None,
    n_zones: int = None
) -> dict:
    """Recompute zones for a single fold."""
    if zone_definitions is None:
        zone_definitions = get_zone_definitions(config)
    if n_zones is None:
        n_zones = get_n_zones(config)

    logger.info(f"\n{'='*60}")
    logger.info(f"Recomputing zones for {method} - Outer Fold {outer_fold}")
    logger.info(f"Zone modes: {zone_modes}")
    logger.info(f"Number of zones: {n_zones}")
    logger.info(f"{'='*60}")

    test_eval_dir = (
        PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
        method / f'outer_fold_{outer_fold}'
    )

    predictions_path = test_eval_dir / "test_predictions.csv"

    if not predictions_path.exists():
        logger.error(f"Predictions not found: {predictions_path}")
        return None

    df = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(df):,} samples from {predictions_path.name}")

    required_cols = ['bootstrap_mean', 'q25', 'q75', 'in_set_1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error("Run add_quantiles.py first to add q25, q75 columns")
        return None

    base_rate = df['target'].mean()
    logger.info(f"  base_rate: {base_rate*100:.3f}% ({base_rate:.4f})")

    y_true = df['target'].values
    results = {'outer_fold': outer_fold, 'n_samples': len(df)}

    for zone_mode in zone_modes:
        logger.info(f"\n--- Zone Mode: {zone_mode.upper()} ---")

        zones = compute_zones_for_mode(df, zone_mode, config)

        logger.info(f"Zone distribution ({zone_mode}):")
        for z in range(n_zones):
            count = (zones == z).sum()
            pct = count / len(df) * 100
            zone_def = zone_definitions.get(z, {})
            logger.info(f"  Zone {z} ({zone_def.get('name', 'N/A')}): {count:,} ({pct:.2f}%)")

        zone_capture = compute_deposit_capture_by_zone(zones, y_true, n_zones=n_zones)

        logger.info(f"Deposit capture by zone ({zone_mode}):")
        for z in range(n_zones):
            stats = zone_capture[f'zone_{z}']
            lift = stats['deposit_pct'] / stats['sample_pct'] if stats['sample_pct'] > 0 else 0
            zone_def = zone_definitions.get(z, {})
            logger.info(
                f"  Zone {z} ({zone_def.get('name', 'N/A')}): "
                f"{stats['n_deposits']} deposits ({stats['deposit_pct']:.1f}%, "
                f"cumulative: {stats['cumulative_deposit_pct']:.1f}%), "
                f"lift={lift:.2f}x"
            )

        practical_zone_dir = test_eval_dir / "practical_zone" / zone_mode
        practical_zone_dir.mkdir(parents=True, exist_ok=True)

        zone_df = df[['index', 'X', 'Y']].copy()
        zone_df['zone'] = zones
        zone_assignments_path = practical_zone_dir / "zone_assignments.csv"
        zone_df.to_csv(zone_assignments_path, index=False)
        logger.info(f"Saved: {zone_assignments_path}")

        n_samples_total = len(df)
        deposit_analysis = {
            'zone_mode': zone_mode,
            'n_zones': n_zones,
            'zone_capture': zone_capture,
            'zone_analysis': {}
        }

        for z in range(n_zones):
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

        validation_criteria = get_validation_criteria(config)
        zone_0_1_min = validation_criteria['zone_0_1_capture_min'] * 100
        zone_0_1_2_min = validation_criteria['zone_0_1_2_capture_min'] * 100
        zone_4_max = validation_criteria['zone_4_capture_max'] * 100

        deposit_analysis['validation'] = {
            'zone_0_1_capture_min': {
                'description': f'Zone 0+1 deposit capture >= {zone_0_1_min:.0f}%',
                'value': zone_capture['zone_1']['cumulative_deposit_pct'],
                'threshold': zone_0_1_min,
                'pass': zone_capture['zone_1']['cumulative_deposit_pct'] >= zone_0_1_min
            },
            'zone_0_1_2_capture_min': {
                'description': f'Zone 0+1+2 deposit capture >= {zone_0_1_2_min:.0f}%',
                'value': zone_capture['zone_2']['cumulative_deposit_pct'],
                'threshold': zone_0_1_2_min,
                'pass': zone_capture['zone_2']['cumulative_deposit_pct'] >= zone_0_1_2_min
            },
            'zone_4_capture_max': {
                'description': f'Zone 4 deposit miss < {zone_4_max:.0f}%',
                'value': zone_capture['zone_4']['deposit_pct'],
                'threshold': zone_4_max,
                'pass': zone_capture['zone_4']['deposit_pct'] < zone_4_max
            }
        }

        deposit_analysis_path = practical_zone_dir / "deposit_analysis.json"
        save_json(deposit_analysis, deposit_analysis_path)
        logger.info(f"Saved: {deposit_analysis_path}")

        results[zone_mode] = {
            'zone_capture': zone_capture,
            'zones': zones
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Recompute zones using Prob/Lift modes"
    )
    parser.add_argument(
        "--method", type=str, default="xgboost",
        help="Method (xgboost, baggingpu_xgboost, tabpfn)"
    )
    parser.add_argument(
        "--outer-fold", type=int, default=None,
        help="Specific outer fold (0-4), or all if not specified"
    )
    parser.add_argument(
        "--zone-mode", type=str, default="all",
        choices=["prob", "lift", "all"],
        help="Zone mode: prob, lift, or all (default: all)"
    )
    args = parser.parse_args()

    config = load_config()

    zone_definitions = get_zone_definitions(config)
    n_zones = get_n_zones(config)

    logger.info(f"Loaded {n_zones} zone definitions from config")

    if args.zone_mode == "all":
        zone_modes = config.get('practical_zone_modes', ['prob', 'lift'])
    else:
        zone_modes = [args.zone_mode]

    logger.info(f"Zone modes to compute: {zone_modes}")

    if args.outer_fold is not None:
        recompute_zones_for_fold(
            args.outer_fold, args.method, config, zone_modes,
            zone_definitions=zone_definitions, n_zones=n_zones
        )
    else:
        logger.info(f"Recomputing zones for all 5 outer folds ({args.method})")
        results = []
        for fold in range(5):
            result = recompute_zones_for_fold(
                fold, args.method, config, zone_modes,
                zone_definitions=zone_definitions, n_zones=n_zones
            )
            if result:
                results.append(result)

        if results:
            logger.info("\n" + "="*70)
            logger.info(f"SUMMARY: Zone Capture Across All Folds")
            logger.info("="*70)

            for zone_mode in zone_modes:
                logger.info(f"\n--- {zone_mode.upper()} Mode ---")

                for z in range(n_zones):
                    zone_def = zone_definitions.get(z, {})
                    deposit_pcts = [r[zone_mode]['zone_capture'][f'zone_{z}']['deposit_pct']
                                   for r in results if zone_mode in r]
                    cum_pcts = [r[zone_mode]['zone_capture'][f'zone_{z}']['cumulative_deposit_pct']
                               for r in results if zone_mode in r]
                    logger.info(
                        f"Zone {z} ({zone_def.get('name', 'N/A')}): "
                        f"deposit={np.mean(deposit_pcts):.1f}% ± {np.std(deposit_pcts):.1f}%, "
                        f"cumulative={np.mean(cum_pcts):.1f}% ± {np.std(cum_pcts):.1f}%"
                    )

                z01_caps = [r[zone_mode]['zone_capture']['zone_1']['cumulative_deposit_pct']
                           for r in results if zone_mode in r]
                logger.info(f"Zone 0+1 capture: {np.mean(z01_caps):.1f}% ± {np.std(z01_caps):.1f}%")

                z4_miss = [r[zone_mode]['zone_capture']['zone_4']['deposit_pct']
                          for r in results if zone_mode in r]
                logger.info(f"Zone 4 miss rate: {np.mean(z4_miss):.1f}% ± {np.std(z4_miss):.1f}%")


if __name__ == "__main__":
    main()
