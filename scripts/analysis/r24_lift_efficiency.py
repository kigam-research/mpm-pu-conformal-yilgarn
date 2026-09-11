#!/usr/bin/env python3
"""R2.4 lift-threshold efficiency + R2.2 paired-difference supplement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from r1324_r22_analysis import (  # noqa: E402
    CURRENT_LIFT,
    MODELS,
    OUTPUT_ROOT,
    TABLE5_COVERAGE,
    TABLE6_TARGETS,
    GateResult,
    bootstrap_ci_over_folds,
    build_zone_sweep,
    compare_snapshots,
    compute_calibration_metrics,
    load_oof_predictions,
    markdown_table,
    rel_path,
    safe_ratio,
    snapshot_protected,
    write_json,
    write_snapshot,
)

EFFICIENCY_CUTOFFS = (0.01, 0.025, 0.05, 0.075, 0.10)

C_CALIBRATED_EXPECTED = {
    "baggingpu_xgboost": {"ece_adaptive": 0.007962, "brier_skill_score": 0.075318},
    "xgboost": {"ece_adaptive": 0.011203, "brier_skill_score": 0.056607},
}


def lift_configs() -> list[dict[str, Any]]:
    """Three zone configs: top tier fixed at 1%, every boundary within top 10%."""
    return [
        {
            "analysis": "B2_lift_config",
            "config_name": "tier_1_2.5_5",
            "rel_iqr_threshold": 0.30,
            "lift_z0": 0.01,
            "lift_z1": 0.025,
            "lift_z2": 0.05,
        },
        {
            "analysis": "B2_lift_config",
            "config_name": "tier_1_5_10_current",
            "rel_iqr_threshold": 0.30,
            "lift_z0": CURRENT_LIFT[0],
            "lift_z1": CURRENT_LIFT[1],
            "lift_z2": CURRENT_LIFT[2],
        },
        {
            "analysis": "B2_lift_config",
            "config_name": "tier_1_2.5_10",
            "rel_iqr_threshold": 0.30,
            "lift_z0": 0.01,
            "lift_z1": 0.025,
            "lift_z2": 0.10,
        },
        {
            "analysis": "B2_lift_config",
            "config_name": "tier_1_7_10",
            "rel_iqr_threshold": 0.30,
            "lift_z0": 0.01,
            "lift_z1": 0.07,
            "lift_z2": 0.10,
        },
    ]


def add_zone012_lift(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["zone012_lift_mean"] = [
        safe_ratio(c, a)
        for c, a in zip(out["zone012_capture_pct_mean"], out["zone012_area_pct_mean"])
    ]
    out["zone01_capture_pct_mean_chk"] = out["zone01_capture_pct_mean"]
    return out


def compute_efficiency_curve(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative and marginal deposit capture-per-area on the prob ranking."""
    fold_rows: list[dict[str, Any]] = []
    for model, df in datasets.items():
        for fold, fold_df in df.groupby("fold", sort=True):
            prob = fold_df["bootstrap_mean"].to_numpy()
            y = fold_df["target"].to_numpy()
            n = len(fold_df)
            total_dep = int(y.sum())
            prev_cap = 0.0
            prev_area = 0.0
            for cutoff in EFFICIENCY_CUTOFFS:
                thresh = float(np.percentile(prob, 100.0 * (1.0 - cutoff)))
                mask = prob >= thresh
                n_top = int(mask.sum())
                n_dep_top = int(y[mask].sum())
                area_pct = 100.0 * n_top / n
                cap_pct = 100.0 * n_dep_top / total_dep if total_dep else float("nan")
                cum_lift = safe_ratio(cap_pct, area_pct)
                marg_area = area_pct - prev_area
                marg_cap = cap_pct - prev_cap
                marg_lift = safe_ratio(marg_cap, marg_area)
                fold_rows.append(
                    {
                        "model": model,
                        "model_label": MODELS[model],
                        "fold": int(fold),
                        "cutoff_pct": cutoff * 100.0,
                        "n_cells_top": n_top,
                        "n_deposits_top": n_dep_top,
                        "area_pct": area_pct,
                        "capture_pct": cap_pct,
                        "cumulative_lift": cum_lift,
                        "marginal_capture_pct": marg_cap,
                        "marginal_area_pct": marg_area,
                        "marginal_lift": marg_lift,
                    }
                )
                prev_cap, prev_area = cap_pct, area_pct
    fold_df = pd.DataFrame(fold_rows)

    summary_rows = []
    metric_cols = [
        "area_pct",
        "capture_pct",
        "cumulative_lift",
        "marginal_capture_pct",
        "marginal_area_pct",
        "marginal_lift",
    ]
    for (model, cutoff), group in fold_df.groupby(["model", "cutoff_pct"], sort=True):
        row = {
            "model": model,
            "model_label": MODELS[model],
            "cutoff_pct": cutoff,
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_sd"] = float(group[col].std(ddof=1))
        summary_rows.append(row)
    return fold_df, pd.DataFrame(summary_rows)


def compute_r22_paired_diff(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, GateResult]:
    """Paired BaggingPU minus XGBoost difference over folds for calibrated probs."""
    c_fold, c_summary, _reliability = compute_calibration_metrics(datasets)

    consistency_ok = True
    details = []
    for model, expected in C_CALIBRATED_EXPECTED.items():
        row = c_summary[
            (c_summary["model"] == model)
            & (c_summary["probability_col"] == "calibrated_prob")
        ].iloc[0]
        for metric, exp_val in expected.items():
            got = float(row[f"{metric}_mean"])
            diff = abs(got - exp_val)
            if diff > 1e-5:
                consistency_ok = False
            details.append(f"{model}.{metric}={got:.6f} exp={exp_val:.6f} d={diff:.2e}")
    consistency_gate = GateResult(
        "Recomputed calibration matches gated C summary",
        consistency_ok,
        " | ".join(details),
    )

    cal = c_fold[c_fold["probability_col"] == "calibrated_prob"]
    rows = []
    for metric in ("ece_adaptive", "brier_skill_score", "brier"):
        xgb = cal[cal["model"] == "xgboost"].sort_values("fold")[metric].to_numpy()
        bpu = (
            cal[cal["model"] == "baggingpu_xgboost"]
            .sort_values("fold")[metric]
            .to_numpy()
        )
        diff = bpu - xgb
        low, high = bootstrap_ci_over_folds(diff.tolist(), n_resamples=1000)
        n_neg = int((diff < 0).sum())
        n_pos = int((diff > 0).sum())
        rows.append(
            {
                "metric": metric,
                "direction": "baggingpu_minus_xgboost (calibrated)",
                "xgb_mean": float(np.mean(xgb)),
                "bpu_mean": float(np.mean(bpu)),
                "mean_diff": float(np.mean(diff)),
                "sd_diff": float(np.std(diff, ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "n_folds_bpu_lower": n_neg,
                "n_folds_bpu_higher": n_pos,
                "per_fold_diff": ";".join(f"{d:.6f}" for d in diff),
                "ci_crosses_zero": bool(low <= 0.0 <= high),
            }
        )
    return c_summary, pd.DataFrame(rows), consistency_gate


def check_table6(b2_summary: pd.DataFrame) -> GateResult:
    cur = b2_summary[b2_summary["config_name"] == "tier_1_5_10_current"]
    passed = True
    out = []
    for model, targets in TABLE6_TARGETS.items():
        row = cur[cur["model"] == model].iloc[0]
        for metric, target in targets.items():
            got = float(row[metric])
            d = abs(got - target)
            if d > 0.10:
                passed = False
            out.append(f"{model}.{metric}={got:.2f}~{target:.2f}(d={d:.3f})")
    return GateResult("Table 6 reproduction (current 1/5/10)", passed, " | ".join(out))


def check_zone4_coverage(b2_summary: pd.DataFrame) -> GateResult:
    cur = b2_summary[b2_summary["config_name"] == "tier_1_5_10_current"]
    passed = True
    out = []
    for model, coverage in TABLE5_COVERAGE.items():
        row = cur[cur["model"] == model].iloc[0]
        z4 = float(row["zone4_capture_pct_mean"])
        exp_miss = 100.0 - coverage
        d = abs(z4 - exp_miss)
        if d > 0.10:
            passed = False
        out.append(f"{model}: zone4={z4:.2f} 100-cov={exp_miss:.2f} d={d:.3f}")
    return GateResult("Zone4 == 1 - positive coverage", passed, " | ".join(out))


def check_efficiency_monotone(eff_summary: pd.DataFrame) -> GateResult:
    passed = True
    out = []
    for model, group in eff_summary.groupby("model", sort=True):
        g = group.sort_values("cutoff_pct")
        cap = g["capture_pct_mean"].to_numpy()
        marg = g["marginal_lift_mean"].to_numpy()
        cap_mono = bool(np.all(np.diff(cap) >= -1e-9))
        marg_first_max = bool(marg[0] >= marg[1:].max() - 1e-9)
        if not cap_mono:
            passed = False
        out.append(
            f"{model}: cap_monotone={cap_mono} cap@10%={cap[-1]:.1f} "
            f"marg_lift={'>'.join(f'{m:.1f}' for m in marg)} first_is_max={marg_first_max}"
        )
    return GateResult("Efficiency cumulative-capture monotone", passed, " | ".join(out))


def make_report(
    b2_summary: pd.DataFrame,
    eff_summary: pd.DataFrame,
    paired: pd.DataFrame,
    gates: list[GateResult],
    artifacts: list[Path],
) -> str:
    lines = ["# R2.4 lift-efficiency + R2.2 paired-difference report", ""]
    lines.append("## Hard gates")
    for g in gates:
        lines.append(f"- {'PASS' if g.passed else 'FAIL'}: {g.name} - {g.details}")
    lines.append("")

    lines.append("## (B2) lift-config zone sweep (top tier fixed 1%, all <= top 10%)")
    b2_cols = [
        "model_label",
        "config_name",
        "zone0_capture_pct_mean",
        "zone0_area_pct_mean",
        "zone01_capture_pct_mean",
        "zone01_area_pct_mean",
        "zone012_capture_pct_mean",
        "zone012_area_pct_mean",
        "zone012_lift_mean",
        "zone4_capture_pct_mean",
    ]
    lines.append(markdown_table(b2_summary, b2_cols, float_digits=4))
    lines.append("")

    lines.append("## (G) probability-ranking efficiency / marginal-lift curve")
    eff_cols = [
        "model_label",
        "cutoff_pct",
        "area_pct_mean",
        "capture_pct_mean",
        "capture_pct_sd",
        "cumulative_lift_mean",
        "marginal_capture_pct_mean",
        "marginal_lift_mean",
    ]
    lines.append(markdown_table(eff_summary, eff_cols, float_digits=4))
    lines.append("")

    lines.append("## (H) R2.2 paired difference (BaggingPU - XGBoost, calibrated)")
    h_cols = [
        "metric",
        "xgb_mean",
        "bpu_mean",
        "mean_diff",
        "sd_diff",
        "ci95_low",
        "ci95_high",
        "n_folds_bpu_lower",
        "n_folds_bpu_higher",
        "ci_crosses_zero",
    ]
    lines.append(markdown_table(paired, h_cols, float_digits=6))
    lines.append("")
    lines.append("Per-fold diffs are in `H_r22_paired_diff.csv`.")
    lines.append("")

    lines.append("## Artifacts")
    for a in artifacts:
        lines.append(f"- {rel_path(a)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    runtime_before = OUTPUT_ROOT / "r24_protected_snapshot_before.tsv"
    write_snapshot(runtime_before, before)

    datasets = {model: load_oof_predictions(model) for model in MODELS}

    b2_fold, b2_summary = build_zone_sweep(datasets, lift_configs())
    b2_summary = add_zone012_lift(b2_summary)

    eff_fold, eff_summary = compute_efficiency_curve(datasets)
    c_summary, paired, consistency_gate = compute_r22_paired_diff(datasets)

    b2_fold.to_csv(OUTPUT_ROOT / "B2_lift_config_sensitivity_fold.csv", index=False)
    b2_summary.to_csv(
        OUTPUT_ROOT / "B2_lift_config_sensitivity_summary.csv", index=False
    )
    eff_fold.to_csv(OUTPUT_ROOT / "G_efficiency_curve_fold.csv", index=False)
    eff_summary.to_csv(OUTPUT_ROOT / "G_efficiency_curve_summary.csv", index=False)
    paired.to_csv(OUTPUT_ROOT / "H_r22_paired_diff.csv", index=False)

    after = snapshot_protected()
    after_path = OUTPUT_ROOT / "r24_protected_snapshot_after.tsv"
    write_snapshot(after_path, after)
    snapshot_ok, snapshot_msgs = compare_snapshots(before, after)

    gates = [
        check_table6(b2_summary),
        check_zone4_coverage(b2_summary),
        consistency_gate,
        check_efficiency_monotone(eff_summary),
        GateResult(
            "Original snapshot unchanged",
            snapshot_ok,
            "; ".join(snapshot_msgs) if snapshot_msgs else f"entries={len(before)}",
        ),
    ]

    write_json(
        OUTPUT_ROOT / "r24_hard_gates.json",
        [{"name": g.name, "passed": g.passed, "details": g.details} for g in gates],
    )

    artifacts = [
        runtime_before,
        after_path,
        OUTPUT_ROOT / "B2_lift_config_sensitivity_fold.csv",
        OUTPUT_ROOT / "B2_lift_config_sensitivity_summary.csv",
        OUTPUT_ROOT / "G_efficiency_curve_fold.csv",
        OUTPUT_ROOT / "G_efficiency_curve_summary.csv",
        OUTPUT_ROOT / "H_r22_paired_diff.csv",
        OUTPUT_ROOT / "r24_hard_gates.json",
        OUTPUT_ROOT / "r24_report.md",
    ]
    report = make_report(b2_summary, eff_summary, paired, gates, artifacts)
    (OUTPUT_ROOT / "r24_report.md").write_text(report, encoding="utf-8")
    print(report)

    if not all(g.passed for g in gates):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
