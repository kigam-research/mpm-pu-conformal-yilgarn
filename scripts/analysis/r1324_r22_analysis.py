#!/usr/bin/env python3
"""Aggregate-only R1.3/R2.4 threshold and R2.2 calibration analysis."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MPM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MPM_ROOT.parent
sys.path.insert(0, str(MPM_ROOT))

from scripts.utils.conformal_utils import (  # noqa: E402
    assign_zones_lift,
    compute_deposit_capture_by_zone,
)

OUTPUT_ROOT = MPM_ROOT / "outputs" / "r1324_r22"

MODELS = {
    "xgboost": "standard XGBoost",
    "baggingpu_xgboost": "BaggingPU-XGBoost",
}

PROTECTED_DIRS = [
    "data",
    "outputs/aggregated",
    "outputs/aggregated_oof",
    "outputs/conformal",
    "outputs/models",
    "outputs/folds",
]

TABLE6_TARGETS = {
    "xgboost": {
        "zone0_capture_pct_mean": 0.73,
        "zone0_area_pct_mean": 0.02,
        "zone012_capture_pct_mean": 54.86,
        "zone012_area_pct_mean": 5.00,
        "zone4_capture_pct_mean": 8.70,
    },
    "baggingpu_xgboost": {
        "zone0_capture_pct_mean": 17.25,
        "zone0_area_pct_mean": 0.68,
        "zone012_capture_pct_mean": 54.98,
        "zone012_area_pct_mean": 5.08,
        "zone4_capture_pct_mean": 9.56,
    },
}

TABLE5_COVERAGE = {
    "xgboost": 91.30,
    "baggingpu_xgboost": 90.44,
}

PROBABILITY_COLUMNS = ("bootstrap_mean", "calibrated_prob")
RHO_THRESHOLDS = (0.20, 0.30, 0.40)
CURRENT_LIFT = (0.01, 0.05, 0.10)
TIGHT_LIFT = (0.005, 0.025, 0.05)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    details: str


def rel_path(path: Path) -> str:
    return str(path.relative_to(MPM_ROOT))


def snapshot_protected() -> dict[str, tuple[str, int, int]]:
    snapshot: dict[str, tuple[str, int, int]] = {}
    for rel_dir in PROTECTED_DIRS:
        base = MPM_ROOT / rel_dir
        if not base.exists():
            raise FileNotFoundError(f"Protected path missing: {base}")
        entries = [base, *base.rglob("*")]
        for entry in entries:
            st = entry.stat()
            entry_type = "d" if entry.is_dir() else "f"
            snapshot[rel_path(entry)] = (
                entry_type,
                int(st.st_size),
                int(st.st_mtime_ns),
            )
    return snapshot


def write_snapshot(path: Path, snapshot: dict[str, tuple[str, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# path\ttype\tsize\tmtime_ns\n")
        for key in sorted(snapshot):
            entry_type, size, mtime_ns = snapshot[key]
            fh.write(f"{key}\t{entry_type}\t{size}\t{mtime_ns}\n")


def compare_snapshots(
    before: dict[str, tuple[str, int, int]],
    after: dict[str, tuple[str, int, int]],
) -> tuple[bool, list[str]]:
    changed = [key for key in before if key not in after or before[key] != after[key]]
    added = [key for key in after if key not in before]
    messages = []
    if changed:
        messages.append(f"changed_or_removed={len(changed)} first={changed[:5]}")
    if added:
        messages.append(f"added={len(added)} first={added[:5]}")
    return not changed and not added, messages


def load_oof_predictions(model: str) -> pd.DataFrame:
    path = MPM_ROOT / "outputs" / "aggregated_oof" / model / "merged_predictions.csv"
    cols = [
        "target",
        "bootstrap_mean",
        "calibrated_prob",
        "q25",
        "q75",
        "rel_iqr",
        "in_set_1",
        "fold",
    ]
    df = pd.read_csv(path, usecols=cols)
    if len(df) != 70_200:
        raise ValueError(f"{model}: expected 70,200 OOF rows, got {len(df):,}")
    if int(df["target"].sum()) != 818:
        raise ValueError(
            f"{model}: expected 818 positives, got {int(df['target'].sum())}"
        )
    folds = sorted(df["fold"].unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"{model}: expected folds 0..4, got {folds}")
    return df


def compute_fold_zone_metrics(
    df: pd.DataFrame,
    rel_iqr_threshold: float,
    lift_z0: float,
    lift_z1: float,
    lift_z2: float,
) -> pd.DataFrame:
    """Compute zone metrics by outer fold using within-fold lift percentiles."""
    rows: list[dict[str, Any]] = []
    for fold, fold_df in df.groupby("fold", sort=True):
        local = fold_df.copy()
        zones = assign_zones_lift(
            local,
            prob_col="bootstrap_mean",
            q25_col="q25",
            q75_col="q75",
            in_set_1_col="in_set_1",
            rel_iqr_threshold=rel_iqr_threshold,
            lift_z0=lift_z0,
            lift_z1=lift_z1,
            lift_z2=lift_z2,
        )
        stats = compute_deposit_capture_by_zone(
            zones=zones,
            y_true=local["target"].to_numpy(),
            n_zones=5,
        )
        row: dict[str, Any] = {
            "fold": int(fold),
            "n_cells": int(stats["total"]["n_samples"]),
            "n_deposits": int(stats["total"]["n_deposits"]),
        }
        cumulative_area = 0.0
        for zone in range(5):
            zstats = stats[f"zone_{zone}"]
            area = float(zstats["sample_pct"])
            capture = float(zstats["deposit_pct"])
            cumulative_area += area
            row[f"zone{zone}_n_cells"] = int(zstats["n_samples"])
            row[f"zone{zone}_n_deposits"] = int(zstats["n_deposits"])
            row[f"zone{zone}_area_pct"] = area
            row[f"zone{zone}_capture_pct"] = capture
            row[f"zone{zone}_cumulative_capture_pct"] = float(
                zstats["cumulative_deposit_pct"]
            )
            row[f"zone{zone}_cumulative_area_pct"] = cumulative_area
        row["zone01_capture_pct"] = row["zone1_cumulative_capture_pct"]
        row["zone012_capture_pct"] = row["zone2_cumulative_capture_pct"]
        row["zone01_area_pct"] = row["zone0_area_pct"] + row["zone1_area_pct"]
        row["zone012_area_pct"] = (
            row["zone0_area_pct"] + row["zone1_area_pct"] + row["zone2_area_pct"]
        )
        row["zone4_miss_pct"] = row["zone4_capture_pct"]
        row["positive_coverage_pct"] = 100.0 - row["zone4_capture_pct"]
        row["zone0_capture_per_area"] = safe_ratio(
            row["zone0_capture_pct"], row["zone0_area_pct"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def mean_sd(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def summarize_zone_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    metric_cols = [
        *[f"zone{z}_capture_pct" for z in range(5)],
        *[f"zone{z}_area_pct" for z in range(5)],
        "zone01_capture_pct",
        "zone012_capture_pct",
        "zone01_area_pct",
        "zone012_area_pct",
        "zone4_miss_pct",
        "positive_coverage_pct",
        "zone0_capture_per_area",
    ]
    for col in metric_cols:
        mean, sd = mean_sd(fold_metrics[col])
        summary[f"{col}_mean"] = mean
        summary[f"{col}_sd"] = sd
    return summary


def build_zone_sweep(
    datasets: dict[str, pd.DataFrame],
    configs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    summary_rows = []
    for model, df in datasets.items():
        for cfg in configs:
            fold_metrics = compute_fold_zone_metrics(
                df,
                rel_iqr_threshold=cfg["rel_iqr_threshold"],
                lift_z0=cfg["lift_z0"],
                lift_z1=cfg["lift_z1"],
                lift_z2=cfg["lift_z2"],
            )
            for key, value in cfg.items():
                fold_metrics[key] = value
            fold_metrics["model"] = model
            fold_metrics["model_label"] = MODELS[model]
            fold_rows.append(fold_metrics)

            summary = summarize_zone_fold_metrics(fold_metrics)
            summary.update(cfg)
            summary["model"] = model
            summary["model_label"] = MODELS[model]
            summary_rows.append(summary)
    return pd.concat(fold_rows, ignore_index=True), pd.DataFrame(summary_rows)


def adaptive_equal_mass_bins(pred: np.ndarray, n_bins: int = 10) -> list[np.ndarray]:
    if len(pred) == 0:
        raise ValueError("Cannot bin an empty array.")
    order = np.argsort(pred, kind="mergesort")
    return [
        idx for idx in np.array_split(order, min(n_bins, len(order))) if len(idx) > 0
    ]


def predictive_entropy(pred: np.ndarray) -> np.ndarray:
    clipped = np.clip(pred, 1e-15, 1.0 - 1e-15)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def murphy_decomposition(
    y_true: np.ndarray,
    pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute adaptive ECE and Murphy terms over equal-mass bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and pred must have the same shape.")

    n = len(y)
    bins = adaptive_equal_mass_bins(p, n_bins=n_bins)
    base_rate = float(y.mean())
    uncertainty = base_rate * (1.0 - base_rate)
    ece = 0.0
    reliability = 0.0
    resolution = 0.0
    brier_murphy_direct = 0.0
    bin_rows = []

    for bin_index, idx in enumerate(bins):
        weight = len(idx) / n
        mean_pred = float(p[idx].mean())
        mean_target = float(y[idx].mean())
        ece += weight * abs(mean_pred - mean_target)
        reliability += weight * (mean_pred - mean_target) ** 2
        resolution += weight * (mean_target - base_rate) ** 2
        brier_murphy_direct += weight * (
            (mean_pred - mean_target) ** 2 + mean_target * (1.0 - mean_target)
        )
        bin_rows.append(
            {
                "bin": bin_index,
                "count": int(len(idx)),
                "mean_pred": mean_pred,
                "mean_target": mean_target,
                "weight": weight,
                "pred_min": float(p[idx].min()),
                "pred_max": float(p[idx].max()),
            }
        )

    brier_individual = float(np.mean((p - y) ** 2))
    brier_from_components = reliability - resolution + uncertainty
    brier_murphy_residual = brier_murphy_direct - brier_from_components
    brier_individual_residual = brier_individual - brier_from_components
    bss = float("nan") if uncertainty == 0 else 1.0 - brier_individual / uncertainty

    metrics = {
        "base_rate": base_rate,
        "ece_adaptive": float(ece),
        "brier": brier_individual,
        "brier_murphy_bin_mean": float(brier_murphy_direct),
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability),
        "brier_skill_score": float(bss),
        "sharpness_mean": float(p.mean()),
        "sharpness_sd": float(p.std(ddof=1)),
        "entropy_mean": float(predictive_entropy(p).mean()),
        "brier_murphy_residual": float(brier_murphy_residual),
        "brier_individual_residual": float(brier_individual_residual),
    }
    return metrics, pd.DataFrame(bin_rows)


def bootstrap_ci_over_folds(
    values: list[float], n_resamples: int = 1000
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.isnan(arr).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.nanmean(sample)))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def compute_calibration_metrics(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    reliability_rows = []
    for model, df in datasets.items():
        for prob_col in PROBABILITY_COLUMNS:
            for fold, fold_df in df.groupby("fold", sort=True):
                metrics, _bins = murphy_decomposition(
                    fold_df["target"].to_numpy(),
                    fold_df[prob_col].to_numpy(),
                    n_bins=10,
                )
                row = {
                    "model": model,
                    "model_label": MODELS[model],
                    "probability_col": prob_col,
                    "fold": int(fold),
                    **metrics,
                }
                fold_rows.append(row)

            for subset_name, subset_df in [
                ("full_grid", df),
                ("in_set_1", df[df["in_set_1"].astype(bool)]),
            ]:
                metrics, bins = murphy_decomposition(
                    subset_df["target"].to_numpy(),
                    subset_df[prob_col].to_numpy(),
                    n_bins=10,
                )
                for _, bin_row in bins.iterrows():
                    reliability_rows.append(
                        {
                            "model": model,
                            "model_label": MODELS[model],
                            "probability_col": prob_col,
                            "subset": subset_name,
                            "scope": "pooled_all_folds",
                            "base_rate": metrics["base_rate"],
                            "bin": int(bin_row["bin"]),
                            "count": int(bin_row["count"]),
                            "mean_pred": float(bin_row["mean_pred"]),
                            "mean_target": float(bin_row["mean_target"]),
                            "pred_min": float(bin_row["pred_min"]),
                            "pred_max": float(bin_row["pred_max"]),
                        }
                    )

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    metric_cols = [
        "base_rate",
        "ece_adaptive",
        "brier",
        "brier_murphy_bin_mean",
        "uncertainty",
        "resolution",
        "reliability",
        "brier_skill_score",
        "sharpness_mean",
        "sharpness_sd",
        "entropy_mean",
        "brier_murphy_residual",
        "brier_individual_residual",
    ]
    for (model, prob_col), group in fold_df.groupby(
        ["model", "probability_col"], sort=True
    ):
        row = {
            "model": model,
            "model_label": MODELS[model],
            "probability_col": prob_col,
            "ci_method": "1000 bootstrap resamples over outer-fold metrics",
        }
        for metric in metric_cols:
            values = group[metric].astype(float).tolist()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
            low, high = bootstrap_ci_over_folds(values, n_resamples=1000)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    return fold_df, pd.DataFrame(summary_rows), pd.DataFrame(reliability_rows)


def compute_rho_resolution(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for model, df in datasets.items():
        for fold, fold_df in df.groupby("fold", sort=True):
            local = fold_df.copy()
            prob = local["bootstrap_mean"].to_numpy()
            rel_iqr = (local["q75"].to_numpy() - local["q25"].to_numpy()) / (
                prob + 1e-10
            )
            local["_rel_iqr_calc"] = rel_iqr
            p99 = np.percentile(prob, 99)
            p95 = np.percentile(prob, 95)
            p90 = np.percentile(prob, 90)
            tiers = {
                "top_1": prob >= p99,
                "top_1_5": (prob >= p95) & (prob < p99),
                "top_5_10": (prob >= p90) & (prob < p95),
            }
            in_set = local["in_set_1"].astype(bool).to_numpy()
            total_cells = len(local)
            total_deposits = int(local["target"].sum())
            for tier_name, tier_mask in tiers.items():
                for uncertainty_group, uncertainty_mask in {
                    "low_rho_le_0.30": rel_iqr <= 0.30,
                    "high_rho_gt_0.30": rel_iqr > 0.30,
                }.items():
                    mask = tier_mask & in_set & uncertainty_mask
                    n_cells = int(mask.sum())
                    n_deposits = int(local.loc[mask, "target"].sum())
                    area_pct = 100.0 * n_cells / total_cells
                    capture_pct = (
                        100.0 * n_deposits / total_deposits
                        if total_deposits > 0
                        else float("nan")
                    )
                    density = n_deposits / n_cells if n_cells else float("nan")
                    fold_rows.append(
                        {
                            "model": model,
                            "model_label": MODELS[model],
                            "fold": int(fold),
                            "tier": tier_name,
                            "uncertainty_group": uncertainty_group,
                            "n_cells": n_cells,
                            "n_deposits": n_deposits,
                            "area_pct": area_pct,
                            "capture_pct": capture_pct,
                            "deposit_density": density,
                            "deposit_density_pct": 100.0 * density
                            if not math.isnan(density)
                            else float("nan"),
                            "capture_per_area": safe_ratio(capture_pct, area_pct),
                            "note": "for R2.3; descriptive only here",
                        }
                    )
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for (model, tier, group_name), group in fold_df.groupby(
        ["model", "tier", "uncertainty_group"], sort=True
    ):
        row = {
            "model": model,
            "model_label": MODELS[model],
            "tier": tier,
            "uncertainty_group": group_name,
            "note": "for R2.3; descriptive only here",
        }
        for metric in [
            "n_cells",
            "n_deposits",
            "area_pct",
            "capture_pct",
            "deposit_density_pct",
            "capture_per_area",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    return fold_df, pd.DataFrame(summary_rows)


def check_table6_reproduction(summary_a: pd.DataFrame) -> GateResult:
    rows = []
    passed = True
    current = summary_a[
        (summary_a["analysis"] == "A_rho_sensitivity")
        & (summary_a["rel_iqr_threshold"] == 0.30)
        & (summary_a["lift_z0"] == CURRENT_LIFT[0])
        & (summary_a["lift_z1"] == CURRENT_LIFT[1])
        & (summary_a["lift_z2"] == CURRENT_LIFT[2])
    ]
    for model, targets in TABLE6_TARGETS.items():
        row = current[current["model"] == model].iloc[0]
        diffs = []
        for metric, target in targets.items():
            actual = float(row[metric])
            diff = abs(actual - target)
            diffs.append(f"{metric}={actual:.4f} target={target:.2f} diff={diff:.4f}")
            if diff > 0.10:
                passed = False
        rows.append(f"{model}: " + "; ".join(diffs))
    return GateResult("Table 6 reproduction <=0.1 pp", passed, " | ".join(rows))


def check_zone4_invariant(summary_a: pd.DataFrame) -> GateResult:
    rows = []
    passed = True
    for model, group in summary_a.groupby("model", sort=True):
        values = group.sort_values("rel_iqr_threshold")[
            "zone4_capture_pct_mean"
        ].to_numpy()
        value_range = float(values.max() - values.min())
        if value_range > 1e-12:
            passed = False
        rows.append(
            f"{model}: values={','.join(f'{v:.10f}' for v in values)} range={value_range:.3g}"
        )
    return GateResult("Zone4 miss invariant across rho sweep", passed, " | ".join(rows))


def check_murphy_residual(calibration_summary: pd.DataFrame) -> GateResult:
    values = calibration_summary["brier_murphy_residual_mean"].abs().to_numpy()
    max_abs = float(np.nanmax(values))
    passed = max_abs < 1e-9
    details = f"max_abs_mean_residual={max_abs:.3e}; individual-brier residuals reported separately"
    return GateResult("Murphy bin-mean Brier identity residual <1e-9", passed, details)


def check_zone4_coverage(summary_a: pd.DataFrame) -> GateResult:
    rows = []
    passed = True
    current = summary_a[
        (summary_a["analysis"] == "A_rho_sensitivity")
        & (summary_a["rel_iqr_threshold"] == 0.30)
        & (summary_a["lift_z0"] == CURRENT_LIFT[0])
        & (summary_a["lift_z1"] == CURRENT_LIFT[1])
        & (summary_a["lift_z2"] == CURRENT_LIFT[2])
    ]
    for model, coverage in TABLE5_COVERAGE.items():
        row = current[current["model"] == model].iloc[0]
        zone4 = float(row["zone4_capture_pct_mean"])
        expected_miss = 100.0 - coverage
        diff = abs(zone4 - expected_miss)
        if diff > 0.10:
            passed = False
        rows.append(
            f"{model}: zone4={zone4:.4f}, 100-coverage={expected_miss:.2f}, diff={diff:.4f}"
        )
    return GateResult(
        "Zone4 capture equals 1 - positive coverage", passed, " | ".join(rows)
    )


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def format_mean_sd(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
    if df.empty:
        return "none"
    view = df.loc[:, columns].copy()
    rendered_rows: list[list[str]] = []
    for _, row in view.iterrows():
        rendered = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{float_digits}f}")
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)

    headers = list(view.columns)
    widths = [len(header) for header in headers]
    for rendered in rendered_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, rendered)]

    def fmt_row(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(values, widths))
            + " |"
        )

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rendered_rows)
    return "\n".join(lines)


def make_report(
    summary_a: pd.DataFrame,
    summary_b: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    rho_summary: pd.DataFrame,
    gates: list[GateResult],
    artifacts: list[Path],
) -> str:
    lines: list[str] = []
    lines.append("# R1.3/R2.4/R2.2 Aggregate Analysis Report")
    lines.append("")
    lines.append("## Hard Gates")
    for gate in gates:
        lines.append(
            f"- {'PASS' if gate.passed else 'FAIL'}: {gate.name} - {gate.details}"
        )
    lines.append("")

    lines.append("## (A) rho_IQR sensitivity")
    a_cols = [
        "model_label",
        "rel_iqr_threshold",
        "zone0_capture_pct_mean",
        "zone0_area_pct_mean",
        "zone012_capture_pct_mean",
        "zone012_area_pct_mean",
        "zone4_capture_pct_mean",
        "zone0_capture_per_area_mean",
    ]
    lines.append(markdown_table(summary_a, a_cols, float_digits=4))
    lines.append("")

    lines.append("## (B) lift-tier sensitivity")
    b_cols = [
        "model_label",
        "config_name",
        "zone0_capture_pct_mean",
        "zone0_area_pct_mean",
        "zone012_capture_pct_mean",
        "zone012_area_pct_mean",
        "zone4_capture_pct_mean",
        "zone0_capture_per_area_mean",
    ]
    lines.append(markdown_table(summary_b, b_cols, float_digits=4))
    lines.append("")

    lines.append("## (C) calibration and uncertainty-quality metrics")
    c_cols = [
        "model_label",
        "probability_col",
        "base_rate_mean",
        "ece_adaptive_mean",
        "ece_adaptive_ci95_low",
        "ece_adaptive_ci95_high",
        "brier_mean",
        "brier_skill_score_mean",
        "reliability_mean",
        "resolution_mean",
        "sharpness_mean_mean",
        "entropy_mean_mean",
        "brier_murphy_residual_mean",
        "brier_individual_residual_mean",
    ]
    lines.append(markdown_table(calibration_summary, c_cols, float_digits=6))
    lines.append("")
    lines.append(
        "Murphy identity residual is checked on bin-mean Brier; individual-cell Brier is reported in `brier_mean`."
    )
    lines.append("")

    lines.append("## (D) rho_IQR-resolution diagnostic")
    d_cols = [
        "model_label",
        "tier",
        "uncertainty_group",
        "n_cells_mean",
        "n_deposits_mean",
        "deposit_density_pct_mean",
        "capture_per_area_mean",
    ]
    lines.append(markdown_table(rho_summary, d_cols, float_digits=4))
    lines.append("")

    lines.append("## Artifacts")
    for artifact in artifacts:
        lines.append(f"- {rel_path(artifact)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    before = snapshot_protected()
    runtime_before_path = OUTPUT_ROOT / "_protected_snapshot_runtime_before.tsv"
    write_snapshot(runtime_before_path, before)

    datasets = {model: load_oof_predictions(model) for model in MODELS}

    configs_a = [
        {
            "analysis": "A_rho_sensitivity",
            "config_name": f"rho_{rho:.2f}_lift_1_5_10",
            "rel_iqr_threshold": rho,
            "lift_z0": CURRENT_LIFT[0],
            "lift_z1": CURRENT_LIFT[1],
            "lift_z2": CURRENT_LIFT[2],
        }
        for rho in RHO_THRESHOLDS
    ]
    a_fold, a_summary = build_zone_sweep(datasets, configs_a)

    gate1 = check_table6_reproduction(a_summary)
    if not gate1.passed:
        a_fold.to_csv(OUTPUT_ROOT / "A_rho_sensitivity_fold_metrics.csv", index=False)
        a_summary.to_csv(OUTPUT_ROOT / "A_rho_sensitivity_summary.csv", index=False)
        after = snapshot_protected()
        write_snapshot(OUTPUT_ROOT / "_protected_snapshot_after.tsv", after)
        snapshot_ok, snapshot_messages = compare_snapshots(before, after)
        gates = [
            gate1,
            GateResult(
                "Original snapshot unchanged",
                snapshot_ok,
                "; ".join(snapshot_messages)
                if snapshot_messages
                else f"entries={len(before)}",
            ),
        ]
        write_json(
            OUTPUT_ROOT / "hard_gates.json",
            [{"name": g.name, "passed": g.passed, "details": g.details} for g in gates],
        )
        report = make_report(
            a_summary,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            gates,
            [
                runtime_before_path,
                OUTPUT_ROOT / "A_rho_sensitivity_fold_metrics.csv",
                OUTPUT_ROOT / "A_rho_sensitivity_summary.csv",
            ],
        )
        (OUTPUT_ROOT / "report.md").write_text(report, encoding="utf-8")
        print(report)
        raise SystemExit(2)

    configs_b = [
        {
            "analysis": "B_lift_tier_sensitivity",
            "config_name": "tight_0.5_2.5_5",
            "rel_iqr_threshold": 0.30,
            "lift_z0": TIGHT_LIFT[0],
            "lift_z1": TIGHT_LIFT[1],
            "lift_z2": TIGHT_LIFT[2],
        },
        {
            "analysis": "B_lift_tier_sensitivity",
            "config_name": "current_1_5_10",
            "rel_iqr_threshold": 0.30,
            "lift_z0": CURRENT_LIFT[0],
            "lift_z1": CURRENT_LIFT[1],
            "lift_z2": CURRENT_LIFT[2],
        },
    ]
    b_fold, b_summary = build_zone_sweep(datasets, configs_b)

    c_fold, c_summary, reliability = compute_calibration_metrics(datasets)
    d_fold, d_summary = compute_rho_resolution(datasets)

    a_fold.to_csv(OUTPUT_ROOT / "A_rho_sensitivity_fold_metrics.csv", index=False)
    a_summary.to_csv(OUTPUT_ROOT / "A_rho_sensitivity_summary.csv", index=False)
    b_fold.to_csv(OUTPUT_ROOT / "B_lift_tier_sensitivity_fold_metrics.csv", index=False)
    b_summary.to_csv(OUTPUT_ROOT / "B_lift_tier_sensitivity_summary.csv", index=False)
    c_fold.to_csv(OUTPUT_ROOT / "C_calibration_metrics_by_fold.csv", index=False)
    c_summary.to_csv(OUTPUT_ROOT / "C_calibration_metrics_summary.csv", index=False)
    reliability.to_csv(OUTPUT_ROOT / "C_reliability_diagram_data.csv", index=False)
    d_fold.to_csv(OUTPUT_ROOT / "D_rho_iqr_resolution_by_fold.csv", index=False)
    d_summary.to_csv(OUTPUT_ROOT / "D_rho_iqr_resolution_summary.csv", index=False)

    after = snapshot_protected()
    after_path = OUTPUT_ROOT / "_protected_snapshot_after.tsv"
    write_snapshot(after_path, after)
    snapshot_ok, snapshot_messages = compare_snapshots(before, after)

    gates = [
        gate1,
        check_zone4_invariant(a_summary),
        check_murphy_residual(c_summary),
        check_zone4_coverage(a_summary),
        GateResult(
            "Original snapshot unchanged",
            snapshot_ok,
            "; ".join(snapshot_messages)
            if snapshot_messages
            else f"entries={len(before)}",
        ),
    ]

    write_json(
        OUTPUT_ROOT / "hard_gates.json",
        [{"name": g.name, "passed": g.passed, "details": g.details} for g in gates],
    )

    artifacts = [
        runtime_before_path,
        after_path,
        OUTPUT_ROOT / "A_rho_sensitivity_fold_metrics.csv",
        OUTPUT_ROOT / "A_rho_sensitivity_summary.csv",
        OUTPUT_ROOT / "B_lift_tier_sensitivity_fold_metrics.csv",
        OUTPUT_ROOT / "B_lift_tier_sensitivity_summary.csv",
        OUTPUT_ROOT / "C_calibration_metrics_by_fold.csv",
        OUTPUT_ROOT / "C_calibration_metrics_summary.csv",
        OUTPUT_ROOT / "C_reliability_diagram_data.csv",
        OUTPUT_ROOT / "D_rho_iqr_resolution_by_fold.csv",
        OUTPUT_ROOT / "D_rho_iqr_resolution_summary.csv",
        OUTPUT_ROOT / "hard_gates.json",
        OUTPUT_ROOT / "report.md",
    ]
    report = make_report(a_summary, b_summary, c_summary, d_summary, gates, artifacts)
    (OUTPUT_ROOT / "report.md").write_text(report, encoding="utf-8")
    print(report)

    if not all(g.passed for g in gates):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
