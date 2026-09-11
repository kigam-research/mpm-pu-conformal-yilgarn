#!/usr/bin/env python3
"""13: Generate Summary Statistics and Validation Criteria"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.utils import (
    load_config,
    load_json,
    save_json
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_validation_criteria(
    summary_stats: dict,
    config: dict
) -> dict:
    """Evaluate validation criteria against summary statistics."""
    criteria = config['validation_criteria']

    min_coverage = summary_stats['coverage']['min']
    n_zones = summary_stats.get('n_zones', 5)

    if n_zones == 6 and 'zone_0_1_capture_min' in criteria:
        mean_zone_0_1_capture = summary_stats['zone_capture'].get('zone_1', {}).get('cumulative_mean', 0)
        mean_zone_0_1_2_capture = summary_stats['zone_capture'].get('zone_2', {}).get('cumulative_mean', 0)
        mean_zone_4_capture = summary_stats['zone_capture'].get('zone_4', {}).get('mean', 0)

        validation_results = {
            'criteria_1_coverage': {
                'name': 'Minimum Coverage >= 85%',
                'threshold': criteria['coverage_min'],
                'value': min_coverage,
                'pass': min_coverage >= criteria['coverage_min']
            },
            'criteria_2_zone_0_1_capture': {
                'name': 'Zone 0+1 Deposit Capture >= 25%',
                'threshold': criteria['zone_0_1_capture_min'] * 100,
                'value': mean_zone_0_1_capture,
                'pass': mean_zone_0_1_capture >= criteria['zone_0_1_capture_min'] * 100
            },
            'criteria_3_zone_0_1_2_capture': {
                'name': 'Zone 0+1+2 Deposit Capture >= 40%',
                'threshold': criteria['zone_0_1_2_capture_min'] * 100,
                'value': mean_zone_0_1_2_capture,
                'pass': mean_zone_0_1_2_capture >= criteria['zone_0_1_2_capture_min'] * 100
            },
            'criteria_4_zone_4_capture': {
                'name': 'Zone 4 Deposit Miss < 10%',
                'threshold': criteria['zone_4_capture_max'] * 100,
                'value': mean_zone_4_capture,
                'pass': mean_zone_4_capture < criteria['zone_4_capture_max'] * 100
            }
        }
    else:
        mean_zone_0_capture = summary_stats['zone_capture'].get('zone_0', {}).get('mean', 0)
        mean_zone_4_capture = summary_stats['zone_capture'].get('zone_4', {}).get('mean', 0)
        mean_cumulative_0_3 = summary_stats['zone_capture'].get('zone_3', {}).get('cumulative_mean', 0)

        validation_results = {
            'criteria_1_coverage': {
                'name': 'Minimum Coverage >= 85%',
                'threshold': criteria['coverage_min'],
                'value': min_coverage,
                'pass': min_coverage >= criteria['coverage_min']
            },
            'criteria_2_zone_0_capture': {
                'name': 'Zone 0 Deposit Capture >= 15%',
                'threshold': criteria.get('zone_0_capture_min', 0.15) * 100,
                'value': mean_zone_0_capture,
                'pass': mean_zone_0_capture >= criteria.get('zone_0_capture_min', 0.15) * 100
            },
            'criteria_3_zone_4_capture': {
                'name': 'Zone 4 Deposit Capture < 10%',
                'threshold': criteria.get('zone_4_capture_max', 0.10) * 100,
                'value': mean_zone_4_capture,
                'pass': mean_zone_4_capture < criteria.get('zone_4_capture_max', 0.10) * 100
            },
            'criteria_4_cumulative_0_3': {
                'name': 'Cumulative Zone 0-3 Capture >= 90%',
                'threshold': criteria.get('cumulative_0_3_min', 0.90) * 100,
                'value': mean_cumulative_0_3,
                'pass': mean_cumulative_0_3 >= criteria.get('cumulative_0_3_min', 0.90) * 100
            }
        }

    all_pass = all(c['pass'] for c in validation_results.values() if isinstance(c, dict))
    validation_results['overall_pass'] = all_pass
    validation_results['n_criteria_passed'] = sum(1 for c in validation_results.values() if isinstance(c, dict) and c.get('pass', False))
    validation_results['n_criteria_total'] = 4

    return validation_results


def load_practical_zone_stats(agg_dir: Path) -> dict:
    """Load prob and lift zone statistics."""
    zone_stats = {}

    for mode in ['prob', 'lift']:
        stats_path = agg_dir / 'practical_zone' / mode / 'summary_statistics.json'
        if stats_path.exists():
            zone_stats[mode] = load_json(stats_path)
        else:
            zone_stats[mode] = None

    return zone_stats


def load_fold_level_data(method: str, config: dict) -> list:
    """Load fold-level metrics for detailed reporting."""
    fold_data = []
    n_folds = config['spatial_cv']['n_outer_folds']
    test_eval_dir = PROJECT_ROOT / config['output_dirs']['test_evaluation'] / method

    for fold_idx in range(n_folds):
        fold_dir = test_eval_dir / f"outer_fold_{fold_idx}"
        metrics_path = fold_dir / "metrics.json"

        if metrics_path.exists():
            metrics = load_json(metrics_path)

            for mode in ['prob', 'lift']:
                pz_path = fold_dir / 'practical_zone' / mode / 'deposit_analysis.json'
                if pz_path.exists():
                    pz_data = load_json(pz_path)
                    metrics[f'pz_{mode}'] = pz_data

            fold_data.append(metrics)
        else:
            fold_data.append(None)

    return fold_data


def generate_executive_summary(method_results: dict, practical_zone_stats: dict, config: dict) -> str:
    """Generate executive summary section."""
    lines = [
        "## A. Executive Summary",
        "",
    ]

    all_pass_methods = []
    fail_methods = []

    for method, (summary, validation) in method_results.items():
        if validation['overall_pass']:
            all_pass_methods.append(method.upper())
        else:
            fail_methods.append(method.upper())

    if all_pass_methods:
        lines.append(f"**Methods Passing All Criteria**: {', '.join(all_pass_methods)}")
    if fail_methods:
        lines.append(f"**Methods Failing Some Criteria**: {', '.join(fail_methods)}")
    lines.append("")

    best_score = -1
    best_combo = None

    for method, pz_stats in practical_zone_stats.items():
        for mode in ['prob', 'lift']:
            if pz_stats.get(mode):
                stats = pz_stats[mode]
                eff = stats.get('efficiency_metrics', {})
                z01 = eff.get('zone_0_1_capture', {}).get('mean', 0)
                z012 = eff.get('zone_0_1_2_capture', {}).get('mean', 0)
                z4 = eff.get('zone_4_miss_rate', {}).get('mean', 100)

                score = z01 + z012 - z4
                if score > best_score:
                    best_score = score
                    best_combo = (method.upper(), mode.upper())

    if best_combo:
        lines.extend([
            f"**Recommended Configuration**: {best_combo[0]} with {best_combo[1]} zone mode",
            "",
        ])

    lines.extend([
        "### Key Performance Indicators",
        "",
        "| Method | Coverage | PR-AUC | FNR | Best Zone 0+1 Capture |",
        "|--------|----------|--------|-----|----------------------|",
    ])

    for method, (summary, _) in method_results.items():
        cov = summary['coverage']['mean']
        pr_auc = summary['pr_auc']['mean']
        fnr = summary['fnr']['mean']

        pz = practical_zone_stats.get(method, {})
        best_z01 = 0
        for mode in ['prob', 'lift']:
            if pz.get(mode):
                z01 = pz[mode].get('efficiency_metrics', {}).get('zone_0_1_capture', {}).get('mean', 0)
                best_z01 = max(best_z01, z01)

        lines.append(f"| {method.upper()} | {cov:.4f} | {pr_auc:.4f} | {fnr:.4f} | {best_z01:.1f}% |")

    lines.extend(["", ""])
    return "\n".join(lines)


def generate_method_comparison(method_results: dict, config: dict) -> str:
    """Generate method comparison section with extended metrics."""
    lines = [
        "## B. Method Comparison",
        "",
        "### B.1 Core Performance Metrics",
        "",
        "| Metric | " + " | ".join([m.upper() for m in method_results.keys()]) + " |",
        "|--------|" + "|".join(["--------" for _ in method_results]) + "|"
    ]

    metrics_to_compare = [
        ('Coverage (mean +/- std)', 'coverage', 'pct4'),
        ('Coverage (min)', 'coverage', 'min4'),
        ('Coverage (max)', 'coverage', 'max4'),
        ('FNR (mean +/- std)', 'fnr', 'pct4'),
        ('FNR (min)', 'fnr', 'min4'),
        ('FNR (max)', 'fnr', 'max4'),
        ('PR-AUC (mean +/- std)', 'pr_auc', 'pct4'),
        ('PR-AUC (min)', 'pr_auc', 'min4'),
        ('PR-AUC (max)', 'pr_auc', 'max4'),
    ]

    for metric_name, metric_key, fmt in metrics_to_compare:
        row_values = []
        for method, (summary, _) in method_results.items():
            stats = summary.get(metric_key, {})
            if fmt == 'pct4':
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                row_values.append(f"{mean_val:.4f} +/- {std_val:.4f}")
            elif fmt == 'min4':
                row_values.append(f"{stats.get('min', 0):.4f}")
            elif fmt == 'max4':
                row_values.append(f"{stats.get('max', 0):.4f}")
        lines.append(f"| {metric_name} | " + " | ".join(row_values) + " |")

    lines.extend(["", ""])

    lines.extend([
        "### B.2 Conformal Calibration Parameters",
        "",
        "| Parameter | " + " | ".join([m.upper() for m in method_results.keys()]) + " |",
        "|-----------|" + "|".join(["--------" for _ in method_results]) + "|"
    ])

    calib_metrics = [
        ('Threshold (mean +/- std)', 'threshold', 'sci'),
        ('Threshold (min)', 'threshold', 'sci_min'),
        ('Threshold (max)', 'threshold', 'sci_max'),
    ]

    for metric_name, metric_key, fmt in calib_metrics:
        row_values = []
        for method, (summary, _) in method_results.items():
            stats = summary.get(metric_key, {})
            if fmt == 'sci':
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                row_values.append(f"{mean_val:.6f} +/- {std_val:.6f}")
            elif fmt == 'sci_min':
                row_values.append(f"{stats.get('min', 0):.6f}")
            elif fmt == 'sci_max':
                row_values.append(f"{stats.get('max', 0):.6f}")
        lines.append(f"| {metric_name} | " + " | ".join(row_values) + " |")

    lines.extend(["", ""])
    return "\n".join(lines)


def generate_zone_mode_comparison(practical_zone_stats: dict, config: dict) -> str:
    """Generate Prob vs Lift zone mode comparison table."""
    lines = [
        "## C. Zone Mode Comparison (Prob vs Lift)",
        "",
        "This section compares the two zone assignment strategies:",
        "- **Prob Mode**: Absolute probability thresholds (prob >= 0.50, 0.25, 0.10)",
        "- **Lift Mode**: Relative lift-based percentiles (Top 1%, 5%, 10%)",
        "",
    ]

    for method, pz_stats in practical_zone_stats.items():
        lines.extend([
            f"### {method.upper()}",
            "",
            "| Metric | Prob Mode | Lift Mode |",
            "|--------|-----------|-----------|",
        ])

        prob_stats = pz_stats.get('prob', {})
        lift_stats = pz_stats.get('lift', {})

        def get_eff(stats, key):
            if not stats:
                return "N/A"
            eff = stats.get('efficiency_metrics', {}).get(key, {})
            if isinstance(eff, dict):
                return f"{eff.get('mean', 0):.2f}% +/- {eff.get('std', 0):.2f}%"
            return f"{eff:.2f}%"

        def get_val(stats, key):
            if not stats:
                return "N/A"
            val = stats.get('validation', {}).get(key, 0)
            return f"{val:.1f}%"

        metrics = [
            ('Zone 0+1 Capture (mean +/- std)', 'zone_0_1_capture', get_eff),
            ('Zone 0+1+2 Capture (mean +/- std)', 'zone_0_1_2_capture', get_eff),
            ('Zone 4 Miss Rate (mean +/- std)', 'zone_4_miss_rate', get_eff),
            ('Z01 Capture Pass Rate (>=25%)', 'z01_pass_rate', get_val),
            ('Z012 Capture Pass Rate (>=40%)', 'z012_pass_rate', get_val),
            ('Z4 Miss Pass Rate (<10%)', 'z4_pass_rate', get_val),
        ]

        for name, key, getter in metrics:
            prob_val = getter(prob_stats, key)
            lift_val = getter(lift_stats, key)
            lines.append(f"| {name} | {prob_val} | {lift_val} |")

        lines.extend(["", ""])

    return "\n".join(lines)


def generate_detailed_zone_statistics(method_results: dict, practical_zone_stats: dict, config: dict) -> str:
    """Generate detailed zone statistics for each method x zone_mode combination."""
    lines = [
        "## D. Detailed Zone Statistics by Method",
        "",
    ]

    for method in method_results.keys():
        pz_stats = practical_zone_stats.get(method, {})

        for mode in ['prob', 'lift']:
            stats = pz_stats.get(mode)
            if not stats:
                continue

            lines.extend([
                f"### {method.upper()} - {mode.upper()} Mode",
                "",
            ])

            lines.extend([
                "**Zone Distribution (% of total samples)**",
                "",
                "| Zone | Mean | Std |",
                "|------|------|-----|",
            ])

            zone_dist = stats.get('zone_distribution', {})
            for z in range(5):
                z_key = f'zone_{z}'
                z_data = zone_dist.get(z_key, {})
                mean = z_data.get('mean', 0)
                std = z_data.get('std', 0)
                lines.append(f"| Zone {z} | {mean:.2f}% | {std:.2f}% |")

            lines.extend(["", ""])

            lines.extend([
                "**Zone Deposit Capture (%)**",
                "",
                "| Zone | Mean | Std | Cumulative Mean | Cumulative Std |",
                "|------|------|-----|-----------------|----------------|",
            ])

            zone_cap = stats.get('zone_capture', {})
            for z in range(5):
                z_key = f'zone_{z}'
                z_data = zone_cap.get(z_key, {})
                mean = z_data.get('mean', 0)
                std = z_data.get('std', 0)
                cum_mean = z_data.get('cumulative_mean', 0)
                cum_std = z_data.get('cumulative_std', 0)
                lines.append(f"| Zone {z} | {mean:.2f}% | {std:.2f}% | {cum_mean:.2f}% | {cum_std:.2f}% |")

            lines.extend(["", ""])

            lines.extend([
                "**Efficiency Metrics**",
                "",
                "| Metric | Mean | Std |",
                "|--------|------|-----|",
            ])

            eff = stats.get('efficiency_metrics', {})
            eff_metrics = [
                ('Zone 0+1 Capture', 'zone_0_1_capture'),
                ('Zone 0+1+2 Capture', 'zone_0_1_2_capture'),
                ('Zone 4 Miss Rate', 'zone_4_miss_rate'),
            ]

            for name, key in eff_metrics:
                data = eff.get(key, {})
                mean = data.get('mean', 0) if isinstance(data, dict) else data
                std = data.get('std', 0) if isinstance(data, dict) else 0
                lines.append(f"| {name} | {mean:.2f}% | {std:.2f}% |")

            lines.extend(["", ""])

            lines.extend([
                "**Validation Pass Rates (across folds)**",
                "",
                "| Criterion | Pass Rate |",
                "|-----------|-----------|",
            ])

            val = stats.get('validation', {})
            val_items = [
                ('Zone 0+1 >= 25%', 'z01_pass_rate'),
                ('Zone 0+1+2 >= 40%', 'z012_pass_rate'),
                ('Zone 4 < 10%', 'z4_pass_rate'),
            ]

            for name, key in val_items:
                rate = val.get(key, 0)
                lines.append(f"| {name} | {rate:.1f}% |")

            lines.extend(["", ""])

    return "\n".join(lines)


def generate_fold_level_summary(method_results: dict, fold_data_all: dict, config: dict) -> str:
    """Generate fold-level detailed summary."""
    lines = [
        "## E. Fold-level Results",
        "",
        "This section provides per-fold breakdown of key metrics.",
        "",
    ]

    n_folds = config['spatial_cv']['n_outer_folds']

    for method, fold_data in fold_data_all.items():
        lines.extend([
            f"### {method.upper()}",
            "",
            "**Core Metrics by Fold**",
            "",
            "| Fold | N Test | N Deposits | Coverage | FNR | PR-AUC | Threshold |",
            "|------|--------|------------|----------|-----|--------|-----------|",
        ])

        for fold_idx, fd in enumerate(fold_data):
            if fd is None:
                lines.append(f"| {fold_idx} | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue

            n_test = fd.get('n_test', 0)
            n_dep = fd.get('n_deposits', 0)
            cov = fd.get('coverage', {}).get('class_1_coverage', 0)
            fnr = fd.get('coverage', {}).get('fnr', 0)
            pr_auc = fd.get('pr_auc', 0)
            thresh = fd.get('calibration', {}).get('threshold', 0)

            lines.append(f"| {fold_idx} | {n_test} | {n_dep} | {cov:.4f} | {fnr:.4f} | {pr_auc:.4f} | {thresh:.6f} |")

        lines.extend(["", ""])

        lines.extend([
            "**Platt Calibration Parameters by Fold**",
            "",
            "| Fold | Platt Coefficient | Platt Intercept |",
            "|------|-------------------|-----------------|",
        ])

        for fold_idx, fd in enumerate(fold_data):
            if fd is None:
                lines.append(f"| {fold_idx} | N/A | N/A |")
                continue

            platt_coef = fd.get('calibration', {}).get('platt_coef', 0)
            platt_int = fd.get('calibration', {}).get('platt_intercept', 0)
            lines.append(f"| {fold_idx} | {platt_coef:.4f} | {platt_int:.4f} |")

        lines.extend(["", ""])

        lines.extend([
            "**Zone Capture by Fold (Prob Mode)**",
            "",
            "| Fold | Zone 0 | Zone 1 | Zone 2 | Zone 3 | Zone 4 | Z01 | Z012 |",
            "|------|--------|--------|--------|--------|--------|-----|------|",
        ])

        for fold_idx, fd in enumerate(fold_data):
            if fd is None or 'pz_prob' not in fd:
                lines.append(f"| {fold_idx} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue

            pz = fd['pz_prob']
            zc = pz.get('zone_capture', {})
            z0 = zc.get('zone_0', {}).get('deposit_pct', 0)
            z1 = zc.get('zone_1', {}).get('deposit_pct', 0)
            z2 = zc.get('zone_2', {}).get('deposit_pct', 0)
            z3 = zc.get('zone_3', {}).get('deposit_pct', 0)
            z4 = zc.get('zone_4', {}).get('deposit_pct', 0)
            eff = pz.get('efficiency_metrics', {})
            z01 = eff.get('zone_0_1_capture', 0)
            z012 = eff.get('zone_0_1_2_capture', 0)

            lines.append(f"| {fold_idx} | {z0:.1f}% | {z1:.1f}% | {z2:.1f}% | {z3:.1f}% | {z4:.1f}% | {z01:.1f}% | {z012:.1f}% |")

        lines.extend(["", ""])

        lines.extend([
            "**Zone Capture by Fold (Lift Mode)**",
            "",
            "| Fold | Zone 0 | Zone 1 | Zone 2 | Zone 3 | Zone 4 | Z01 | Z012 |",
            "|------|--------|--------|--------|--------|--------|-----|------|",
        ])

        for fold_idx, fd in enumerate(fold_data):
            if fd is None or 'pz_lift' not in fd:
                lines.append(f"| {fold_idx} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue

            pz = fd['pz_lift']
            zc = pz.get('zone_capture', {})
            z0 = zc.get('zone_0', {}).get('deposit_pct', 0)
            z1 = zc.get('zone_1', {}).get('deposit_pct', 0)
            z2 = zc.get('zone_2', {}).get('deposit_pct', 0)
            z3 = zc.get('zone_3', {}).get('deposit_pct', 0)
            z4 = zc.get('zone_4', {}).get('deposit_pct', 0)
            eff = pz.get('efficiency_metrics', {})
            z01 = eff.get('zone_0_1_capture', 0)
            z012 = eff.get('zone_0_1_2_capture', 0)

            lines.append(f"| {fold_idx} | {z0:.1f}% | {z1:.1f}% | {z2:.1f}% | {z3:.1f}% | {z4:.1f}% | {z01:.1f}% | {z012:.1f}% |")

        lines.extend(["", ""])

    return "\n".join(lines)


def generate_visualization_references(config: dict) -> str:
    """Generate visualization file references section."""
    figures_base = f"outputs/figures"

    lines = [
        "## F. Visualization References",
        "",
        "All generated figures are stored in the `outputs/figures/` directory.",
        "",
        "### CV Metrics",
        f"- Coverage comparison: `{figures_base}/cv_metrics/fold_comparison_coverage.png`",
        f"- FNR comparison: `{figures_base}/cv_metrics/fold_comparison_fnr.png`",
        f"- PR-AUC comparison: `{figures_base}/cv_metrics/fold_comparison_prauc.png`",
        "",
        "### Zone Analysis (Prob Mode)",
        f"- Deposit capture curve: `{figures_base}/zone_analysis/prob/deposit_capture_curve.png`",
        f"- Method comparison: `{figures_base}/zone_analysis/prob/method_comparison_capture.png`",
        f"- Zone distribution (XGBoost): `{figures_base}/zone_analysis/prob/zone_distribution_by_fold_xgboost.png`",
        f"- Zone distribution (BaggingPU): `{figures_base}/zone_analysis/prob/zone_distribution_by_fold_baggingpu_xgboost.png`",
        f"- Zone consistency (XGBoost): `{figures_base}/zone_analysis/prob/zone_consistency_heatmap_xgboost.png`",
        f"- Zone consistency (BaggingPU): `{figures_base}/zone_analysis/prob/zone_consistency_heatmap_baggingpu_xgboost.png`",
        f"- Deposit capture by zone (XGBoost): `{figures_base}/zone_analysis/prob/deposit_capture_by_zone_xgboost.png`",
        f"- Deposit capture by zone (BaggingPU): `{figures_base}/zone_analysis/prob/deposit_capture_by_zone_baggingpu_xgboost.png`",
        "",
        "### Zone Analysis (Lift Mode)",
        f"- Deposit capture curve: `{figures_base}/zone_analysis/lift/deposit_capture_curve.png`",
        f"- Method comparison: `{figures_base}/zone_analysis/lift/method_comparison_capture.png`",
        f"- Zone distribution (XGBoost): `{figures_base}/zone_analysis/lift/zone_distribution_by_fold_xgboost.png`",
        f"- Zone distribution (BaggingPU): `{figures_base}/zone_analysis/lift/zone_distribution_by_fold_baggingpu_xgboost.png`",
        f"- Zone consistency (XGBoost): `{figures_base}/zone_analysis/lift/zone_consistency_heatmap_xgboost.png`",
        f"- Zone consistency (BaggingPU): `{figures_base}/zone_analysis/lift/zone_consistency_heatmap_baggingpu_xgboost.png`",
        f"- Deposit capture by zone (XGBoost): `{figures_base}/zone_analysis/lift/deposit_capture_by_zone_xgboost.png`",
        f"- Deposit capture by zone (BaggingPU): `{figures_base}/zone_analysis/lift/deposit_capture_by_zone_baggingpu_xgboost.png`",
        "",
        "### Spatial Maps",
        f"- Prob mode maps: `{figures_base}/spatial_maps/prob/`",
        f"- Lift mode maps: `{figures_base}/spatial_maps/lift/`",
        "",
        "### Block Examples",
        f"- XGBoost Prob mode: `{figures_base}/block_examples/xgboost/prob/`",
        f"- XGBoost Lift mode: `{figures_base}/block_examples/xgboost/lift/`",
        f"- BaggingPU Prob mode: `{figures_base}/block_examples/baggingpu_xgboost/prob/`",
        f"- BaggingPU Lift mode: `{figures_base}/block_examples/baggingpu_xgboost/lift/`",
        "",
    ]

    return "\n".join(lines)


def generate_computation_sources() -> str:
    """Generate computation source reference table."""
    lines = [
        "## G. Computation Source Reference",
        "",
        "This table maps each computed value to its generating code.",
        "",
        "| Computed Value | Source Code | Description |",
        "|----------------|-------------|-------------|",
        "| Bootstrap predictions | `scripts/bootstrap/train_xgboost_bootstrap.py` | 50 bootstrap models for uncertainty |",
        "| Bootstrap predictions (BaggingPU) | `scripts/bootstrap/train_baggingpu_xgboost_bootstrap.py` | 50 bootstrap BaggingPU models |",
        "| Platt calibration params | `scripts/conformal/cross_conformal_xgboost.py` | Cross-conformal calibration (20 iter) |",
        "| Platt calibration params (BaggingPU) | `scripts/conformal/cross_conformal_baggingpu.py` | Cross-conformal calibration (20 iter) |",
        "| Prob Zone assignments | `scripts/test_evaluation/recompute_zones.py` | `assign_zones_jorc30()` function |",
        "| Lift Zone assignments | `scripts/test_evaluation/recompute_zones.py` | `assign_zones_lift()` function |",
        "| Aggregated statistics | `scripts/result_aggregation/aggregate_fold_results.py` | Fold-level to summary statistics |",
        "| Validation criteria | `scripts/result_aggregation/generate_summary_statistics.py` | This script |",
        "| CV metric figures | `scripts/visualization/plot_cv_metrics.py` | Coverage, FNR, PR-AUC plots |",
        "| Zone analysis figures | `scripts/visualization/plot_practical_zone_analysis.py` | Zone distribution/capture plots |",
        "| Spatial maps | `scripts/visualization/plot_spatial_maps.py` | Geographic zone visualization |",
        "",
    ]

    return "\n".join(lines)


def generate_zone_definitions(config: dict) -> str:
    """Generate zone definitions section."""
    lines = [
        "## H. Zone Definitions",
        "",
        "### Prob Mode (assign_zones_jorc30)",
        "",
        "| Zone | Name | Condition |",
        "|------|------|-----------|",
        "| Zone 0 | IMMEDIATE | prob >= 0.50 AND rel_IQR <= 30% |",
        "| Zone 1 | PRIORITY | (prob >= 0.50 AND rel_IQR > 30%) OR (0.25 <= prob < 0.50 AND rel_IQR <= 30%) |",
        "| Zone 2 | FOLLOW_UP | (0.25 <= prob < 0.50 AND rel_IQR > 30%) OR (0.10 <= prob < 0.25 AND rel_IQR <= 30%) |",
        "| Zone 3 | POTENTIAL | (0.10 <= prob < 0.25 AND rel_IQR > 30%) OR (prob < 0.10 AND in_set_1 = 1) |",
        "| Zone 4 | EXCLUDED | in_set_1 = 0 |",
        "",
        "### Lift Mode (assign_zones_lift)",
        "",
        "| Zone | Name | Condition |",
        "|------|------|-----------|",
        "| Zone 0 | IMMEDIATE | Top 1% (P99) AND rel_IQR <= 30% |",
        "| Zone 1 | PRIORITY | (Top 1% AND rel_IQR > 30%) OR (Top 1-5% AND rel_IQR <= 30%) |",
        "| Zone 2 | FOLLOW_UP | (Top 1-5% AND rel_IQR > 30%) OR (Top 5-10% AND rel_IQR <= 30%) |",
        "| Zone 3 | POTENTIAL | (Top 5-10% AND rel_IQR > 30%) OR (Below Top 10% AND in_set_1 = 1) |",
        "| Zone 4 | EXCLUDED | in_set_1 = 0 |",
        "",
        "### Uncertainty Threshold",
        "",
        "- **rel_IQR**: Relative Interquartile Range = (Q75 - Q25) / bootstrap_mean",
        "- **30% threshold**: Indicated resource-level uncertainty requirement (rel_IQR <= 30%)",
        "",
    ]

    return "\n".join(lines)


def generate_validation_and_recommendations(method_results: dict, practical_zone_stats: dict, config: dict) -> str:
    """Generate validation criteria and recommendations section."""
    lines = [
        "## I. Validation Criteria & Recommendations",
        "",
        "### Validation Criteria (from config.yaml)",
        "",
        "| Criterion | Threshold | Description |",
        "|-----------|-----------|-------------|",
        "| Coverage | >= 85% | Minimum conformal coverage across folds |",
        "| Zone 0+1 Capture | >= 25% | High-priority zones must capture at least 25% of deposits |",
        "| Zone 0+1+2 Capture | >= 40% | Exploration zones must capture at least 40% of deposits |",
        "| Zone 4 Miss Rate | < 10% | Excluded zone must contain less than 10% of deposits |",
        "",
    ]

    for method, (summary, validation) in method_results.items():
        overall_status = "PASS" if validation['overall_pass'] else "FAIL"
        lines.extend([
            f"### {method.upper()} Validation",
            "",
            f"**Overall Status**: {overall_status} ({validation['n_criteria_passed']}/{validation['n_criteria_total']} criteria)",
            "",
            "| Criterion | Threshold | Value | Status |",
            "|-----------|-----------|-------|--------|",
        ])

        for key, crit in validation.items():
            if isinstance(crit, dict) and 'name' in crit:
                status = "PASS" if crit['pass'] else "FAIL"
                thresh_str = f"{crit['threshold']:.2f}" if crit['threshold'] < 2 else f"{crit['threshold']:.1f}%"
                val_str = f"{crit['value']:.4f}" if crit['value'] < 2 else f"{crit['value']:.2f}%"
                lines.append(f"| {crit['name']} | {thresh_str} | {val_str} | {status} |")

        lines.extend(["", ""])

    lines.extend([
        "### Practical Zone Validation Results",
        "",
    ])

    for method, pz_stats in practical_zone_stats.items():
        lines.append(f"**{method.upper()}**")
        lines.append("")

        for mode in ['prob', 'lift']:
            stats = pz_stats.get(mode)
            if not stats:
                continue

            val = stats.get('validation', {})
            z01_pass = val.get('z01_pass_rate', 0)
            z012_pass = val.get('z012_pass_rate', 0)
            z4_pass = val.get('z4_pass_rate', 0)

            overall = "PASS" if (z01_pass >= 60 and z012_pass >= 60 and z4_pass >= 80) else "FAIL"

            lines.extend([
                f"- **{mode.upper()} Mode**: {overall}",
                f"  - Zone 0+1 >= 25%: {z01_pass:.0f}% folds pass",
                f"  - Zone 0+1+2 >= 40%: {z012_pass:.0f}% folds pass",
                f"  - Zone 4 < 10%: {z4_pass:.0f}% folds pass",
            ])

        lines.append("")

    lines.extend([
        "### Recommendations",
        "",
    ])

    best_method = None
    best_mode = None
    best_score = -float('inf')

    for method, pz_stats in practical_zone_stats.items():
        for mode in ['prob', 'lift']:
            stats = pz_stats.get(mode)
            if not stats:
                continue

            val = stats.get('validation', {})
            eff = stats.get('efficiency_metrics', {})

            z01_pass = val.get('z01_pass_rate', 0)
            z012_pass = val.get('z012_pass_rate', 0)
            z4_pass = val.get('z4_pass_rate', 0)
            z01_cap = eff.get('zone_0_1_capture', {}).get('mean', 0)
            z012_cap = eff.get('zone_0_1_2_capture', {}).get('mean', 0)

            score = z01_pass + z012_pass + z4_pass + z01_cap + z012_cap

            if score > best_score:
                best_score = score
                best_method = method
                best_mode = mode

    any_fail = any(not v['overall_pass'] for _, (_, v) in method_results.items())

    if any_fail:
        lines.extend([
            "**Some validation criteria were not met.**",
            "",
            "Consider the following:",
            "- Review zone threshold parameters",
            "- Check for data quality issues",
            "- Analyze fold-specific failures",
            "",
        ])
    else:
        lines.extend([
            "**All core validation criteria passed.**",
            "",
        ])

    if best_method and best_mode:
        lines.extend([
            f"**Recommended Configuration**: {best_method.upper()} with {best_mode.upper()} zone mode",
            "",
            "This configuration provides the best balance of:",
            "- Deposit capture in high-priority zones",
            "- Minimal loss in excluded zones",
            "- Consistent performance across folds",
            "",
        ])

    return "\n".join(lines)


def generate_summary_report(
    method_results: dict,
    practical_zone_stats: dict,
    fold_data_all: dict,
    config: dict
) -> str:
    """Generate comprehensive markdown summary report."""
    report_lines = [
        "# Nested CV Validation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report summarizes the Nested Spatial Block CV validation results for",
        "the MPM (Mineral Prospectivity Mapping) uncertainty quantification framework.",
        "",
        "### Workflow Summary",
        "",
        "1. **Data Preparation**: 5-fold spatial block CV with 50km blocks",
        "2. **Model Training**: XGBoost and BaggingPU(XGBoost) with Optuna optimization",
        "3. **Uncertainty Quantification**: 50 bootstrap models per fold",
        "4. **Conformal Calibration**: Cross-conformal prediction (20 iterations, alpha=0.15)",
        "5. **Zone Classification**: Prob mode (absolute thresholds) and Lift mode (relative percentiles)",
        "6. **Validation**: Coverage, zone capture, and deposit miss rate evaluation",
        "",
        "---",
        "",
    ]

    report_lines.append(generate_executive_summary(method_results, practical_zone_stats, config))
    report_lines.append("---\n\n")

    report_lines.append(generate_method_comparison(method_results, config))
    report_lines.append("---\n\n")

    report_lines.append(generate_zone_mode_comparison(practical_zone_stats, config))
    report_lines.append("---\n\n")

    report_lines.append(generate_detailed_zone_statistics(method_results, practical_zone_stats, config))
    report_lines.append("---\n\n")

    report_lines.append(generate_fold_level_summary(method_results, fold_data_all, config))
    report_lines.append("---\n\n")

    report_lines.append(generate_visualization_references(config))
    report_lines.append("---\n\n")

    report_lines.append(generate_computation_sources())
    report_lines.append("---\n\n")

    report_lines.append(generate_zone_definitions(config))
    report_lines.append("---\n\n")

    report_lines.append(generate_validation_and_recommendations(method_results, practical_zone_stats, config))

    return "\n".join(report_lines)


def generate_summary_statistics(
    method: str = None,
    config: dict = None
):
    """Generate summary statistics and validation criteria."""
    logger.info("=" * 70)
    logger.info("Generating Summary Statistics")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    methods = [method] if method else config['methods']
    method_results = {}
    practical_zone_stats = {}
    fold_data_all = {}

    for m in methods:
        logger.info(f"\n--- Method: {m.upper()} ---")

        agg_dir = PROJECT_ROOT / config['output_dirs']['aggregated'] / m
        summary_path = agg_dir / "summary_statistics.json"

        if not summary_path.exists():
            logger.warning(f"Summary statistics not found: {summary_path}")
            logger.warning("Run aggregate_fold_results.py first!")
            continue

        summary_stats = load_json(summary_path)

        pz_stats = load_practical_zone_stats(agg_dir)
        practical_zone_stats[m] = pz_stats

        fold_data = load_fold_level_data(m, config)
        fold_data_all[m] = fold_data

        validation = evaluate_validation_criteria(summary_stats, config)

        logger.info(f"\nValidation Criteria Results:")
        for key, crit in validation.items():
            if isinstance(crit, dict) and 'name' in crit:
                status = "PASS" if crit['pass'] else "FAIL"
                logger.info(f"  {crit['name']}: {crit['value']:.2f} vs {crit['threshold']} [{status}]")

        overall = "PASS" if validation['overall_pass'] else "FAIL"
        logger.info(f"\nOverall: {overall} ({validation['n_criteria_passed']}/{validation['n_criteria_total']})")

        save_json(validation, agg_dir / "validation_criteria.json")

        method_results[m] = (summary_stats, validation)

    if len(method_results) == len(config['methods']):
        logger.info("\n" + "=" * 70)
        logger.info("Generating Summary Report")
        logger.info("=" * 70)

        report = generate_summary_report(
            method_results,
            practical_zone_stats,
            fold_data_all,
            config
        )

        reports_dir = PROJECT_ROOT / config['output_dirs']['reports']
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "nested_cv_validation_report.md"

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to: {report_path}")

    logger.info("\nSummary statistics generation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary statistics and validation criteria"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['xgboost', 'baggingpu_xgboost'],
        help="Model method (default: all methods)"
    )

    args = parser.parse_args()
    generate_summary_statistics(args.method)


if __name__ == "__main__":
    main()
