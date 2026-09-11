#!/usr/bin/env python3
"""R2.2 supplement: decision-region (conformal in-set) calibration quality."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from r1324_r22_analysis import (  # noqa: E402
    MODELS,
    OUTPUT_ROOT,
    GateResult,
    bootstrap_ci_over_folds,
    compare_snapshots,
    load_oof_predictions,
    markdown_table,
    murphy_decomposition,
    snapshot_protected,
    write_json,
    write_snapshot,
)

PROBABILITY_COLUMNS = ("bootstrap_mean", "calibrated_prob")

C_FULLGRID_EXPECTED = {
    "baggingpu_xgboost": {"ece_adaptive": 0.007962, "brier_skill_score": 0.075318},
    "xgboost": {"ece_adaptive": 0.011203, "brier_skill_score": 0.056607},
}


def compute_subset_calibration(
    datasets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold ECE/Brier/BSS/base-rate on full grid and on the conformal in-set."""
    fold_rows: list[dict[str, Any]] = []
    for model, df in datasets.items():
        for prob_col in PROBABILITY_COLUMNS:
            for subset_name, mask_fn in (
                ("full_grid", lambda d: np.ones(len(d), dtype=bool)),
                ("in_set_1", lambda d: d["in_set_1"].astype(bool).to_numpy()),
            ):
                for fold, fold_df in df.groupby("fold", sort=True):
                    mask = mask_fn(fold_df)
                    sub = fold_df.loc[mask]
                    y = sub["target"].to_numpy()
                    p = sub[prob_col].to_numpy()
                    n = len(sub)
                    n_pos = int(y.sum())
                    if n_pos == 0 or n_pos == n:
                        metrics = {
                            "base_rate": float(y.mean()) if n else float("nan"),
                            "ece_adaptive": float("nan"),
                            "brier": float("nan"),
                            "brier_skill_score": float("nan"),
                        }
                    else:
                        metrics, _ = murphy_decomposition(y, p, n_bins=10)
                    fold_rows.append(
                        {
                            "model": model,
                            "model_label": MODELS[model],
                            "probability_col": prob_col,
                            "subset": subset_name,
                            "fold": int(fold),
                            "n_cells": n,
                            "n_deposits": n_pos,
                            "base_rate_pct": 100.0 * n_pos / n if n else float("nan"),
                            "ece_adaptive": metrics["ece_adaptive"],
                            "brier": metrics["brier"],
                            "brier_skill_score": metrics["brier_skill_score"],
                        }
                    )
    fold_df = pd.DataFrame(fold_rows)

    summary_rows = []
    metric_cols = [
        "n_cells",
        "n_deposits",
        "base_rate_pct",
        "ece_adaptive",
        "brier",
        "brier_skill_score",
    ]
    for (model, prob_col, subset), group in fold_df.groupby(
        ["model", "probability_col", "subset"], sort=True
    ):
        row = {
            "model": model,
            "model_label": MODELS[model],
            "probability_col": prob_col,
            "subset": subset,
        }
        for col in metric_cols:
            vals = group[col].astype(float).tolist()
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_sd"] = float(np.nanstd(vals, ddof=1))
            if col in ("ece_adaptive", "brier_skill_score"):
                low, high = bootstrap_ci_over_folds(vals, n_resamples=1000)
                row[f"{col}_ci95_low"] = low
                row[f"{col}_ci95_high"] = high
        summary_rows.append(row)
    return fold_df, pd.DataFrame(summary_rows)


def compute_inset_paired(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Paired BaggingPU minus XGBoost diff for in-set calibrated ECE/BSS."""
    cal = fold_df[
        (fold_df["probability_col"] == "calibrated_prob")
        & (fold_df["subset"] == "in_set_1")
    ]
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
        rows.append(
            {
                "metric": metric,
                "subset": "in_set_1",
                "xgb_mean": float(np.mean(xgb)),
                "bpu_mean": float(np.mean(bpu)),
                "mean_diff": float(np.mean(diff)),
                "sd_diff": float(np.std(diff, ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "n_folds_bpu_lower": int((diff < 0).sum()),
                "n_folds_bpu_higher": int((diff > 0).sum()),
                "per_fold_diff": ";".join(f"{d:.6f}" for d in diff),
                "ci_crosses_zero": bool(low <= 0.0 <= high),
            }
        )
    return pd.DataFrame(rows)


def consistency_gate(summary: pd.DataFrame) -> GateResult:
    ok = True
    details = []
    for model, expected in C_FULLGRID_EXPECTED.items():
        row = summary[
            (summary["model"] == model)
            & (summary["probability_col"] == "calibrated_prob")
            & (summary["subset"] == "full_grid")
        ].iloc[0]
        for metric, exp_val in expected.items():
            got = float(row[f"{metric}_mean"])
            d = abs(got - exp_val)
            if d > 1e-5:
                ok = False
            details.append(f"{model}.{metric}={got:.6f} exp={exp_val:.6f} d={d:.2e}")
    return GateResult(
        "Full-grid calibrated matches gated C summary", ok, " | ".join(details)
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    before = snapshot_protected()
    before_path = OUTPUT_ROOT / "r22inset_protected_snapshot_before.tsv"
    write_snapshot(before_path, before)

    datasets = {model: load_oof_predictions(model) for model in MODELS}
    fold_df, summary = compute_subset_calibration(datasets)
    paired = compute_inset_paired(fold_df)

    fold_df.to_csv(OUTPUT_ROOT / "r22inset_calibration_by_fold.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "r22inset_calibration_summary.csv", index=False)
    paired.to_csv(OUTPUT_ROOT / "r22inset_paired_diff.csv", index=False)

    after = snapshot_protected()
    after_path = OUTPUT_ROOT / "r22inset_protected_snapshot_after.tsv"
    write_snapshot(after_path, after)
    snap_ok, snap_msgs = compare_snapshots(before, after)

    gates = [
        consistency_gate(summary),
        GateResult(
            "Original snapshot unchanged",
            snap_ok,
            "; ".join(snap_msgs) if snap_msgs else f"entries={len(before)}",
        ),
    ]
    write_json(
        OUTPUT_ROOT / "r22inset_hard_gates.json",
        [{"name": g.name, "passed": g.passed, "details": g.details} for g in gates],
    )

    lines = ["# R2.2 in-set (decision-region) calibration", ""]
    lines.append("## Hard gates")
    for g in gates:
        lines.append(f"- {'PASS' if g.passed else 'FAIL'}: {g.name} - {g.details}")
    lines.append("")
    lines.append("## Calibration by subset (means across 5 folds)")
    cols = [
        "model_label",
        "probability_col",
        "subset",
        "n_cells_mean",
        "base_rate_pct_mean",
        "ece_adaptive_mean",
        "ece_adaptive_ci95_low",
        "ece_adaptive_ci95_high",
        "brier_skill_score_mean",
    ]
    lines.append(markdown_table(summary, cols, float_digits=4))
    lines.append("")
    lines.append("## Paired BaggingPU - XGBoost, in-set calibrated")
    pcols = [
        "metric",
        "xgb_mean",
        "bpu_mean",
        "mean_diff",
        "ci95_low",
        "ci95_high",
        "n_folds_bpu_lower",
        "n_folds_bpu_higher",
        "ci_crosses_zero",
    ]
    lines.append(markdown_table(paired, pcols, float_digits=6))
    lines.append("")
    report = "\n".join(lines)
    (OUTPUT_ROOT / "r22inset_report.md").write_text(report, encoding="utf-8")
    print(report)

    if not all(g.passed for g in gates):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
