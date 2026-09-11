#!/usr/bin/env python3
"""R2.6 uncertainty-triage numbers from frozen OOF predictions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MPM_ROOT = Path(__file__).resolve().parents[2]
OUT = MPM_ROOT / "outputs" / "r1324_r22"

MODELS = {
    "baggingpu_xgboost": "BaggingPU-XGBoost",
    "xgboost": "standard XGBoost",
}
REQUIRED_COLUMNS = ["bootstrap_mean", "fold", "in_set_1", "rel_iqr", "target"]
RHO_CUT = 0.30
D_SUMMARY_TOL = 0.05
R23_CAPTURE_TARGET = {"baggingpu_xgboost": 33.0, "xgboost": 23.0}
R23_CAPTURE_TOL_PP = 0.5


def input_path(model: str) -> Path:
    return MPM_ROOT / "outputs" / "aggregated_oof" / model / "merged_predictions.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_oof(model: str) -> pd.DataFrame:
    df = pd.read_csv(input_path(model), usecols=REQUIRED_COLUMNS)
    if len(df) != 70_200:
        raise ValueError(f"{model}: expected 70,200 rows, got {len(df):,}")
    if int(df["target"].sum()) != 818:
        raise ValueError(f"{model}: expected 818 deposits")
    folds = sorted(df["fold"].unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"{model}: expected folds 0..4, got {folds}")
    return df


def frac(numerator: float, denominator: float) -> float:
    return math.nan if denominator == 0 else float(numerator) / float(denominator)


def corr_or_nan(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return math.nan
    if (
        float(df["rel_iqr"].std(ddof=1)) == 0
        or float(df["bootstrap_mean"].std(ddof=1)) == 0
    ):
        return math.nan
    return float(df["rel_iqr"].corr(df["bootstrap_mean"]))


def tier_masks(fold_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    probs = fold_df["bootstrap_mean"].to_numpy()
    p99 = float(np.percentile(probs, 99))
    p95 = float(np.percentile(probs, 95))
    top1 = fold_df["bootstrap_mean"] >= p99
    top1_5 = (fold_df["bootstrap_mean"] >= p95) & (fold_df["bootstrap_mean"] < p99)
    return top1, top1_5


def compute_model(model: str, label: str, df: pd.DataFrame) -> tuple[list, list, list]:
    split_rows: list[dict[str, Any]] = []
    r23_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for fold, fold_df in df.groupby("fold", sort=True):
        top1, top1_5 = tier_masks(fold_df)
        in_set = fold_df["in_set_1"].astype(bool)
        low = fold_df["rel_iqr"] <= RHO_CUT
        high = fold_df["rel_iqr"] > RHO_CUT
        top1_set = fold_df[top1 & in_set]
        total_cells = int(len(top1_set))
        total_deposits = int(top1_set["target"].sum())

        groups = {
            "immediate_low_rho_le_0.30": top1 & in_set & low,
            "additional_evidence_high_rho_gt_0.30": top1 & in_set & high,
            "total_top1_in_set_1": top1 & in_set,
        }
        for group_name, mask in groups.items():
            part = fold_df[mask]
            n_cells = int(len(part))
            n_deposits = int(part["target"].sum())
            split_rows.append(
                {
                    "model": model,
                    "model_label": label,
                    "scope": "fold",
                    "fold": int(fold),
                    "tier": "top_1",
                    "probability_col": "bootstrap_mean",
                    "top1_rule": "per_fold_prob_ge_p99_and_in_set_1",
                    "triage_group": group_name,
                    "n_cells": n_cells,
                    "n_cells_sd": math.nan,
                    "n_deposits": n_deposits,
                    "n_deposits_sd": math.nan,
                    "cell_fraction_of_top1": 1.0
                    if group_name == "total_top1_in_set_1"
                    else frac(n_cells, total_cells),
                    "deposit_fraction_of_top1": 1.0
                    if group_name == "total_top1_in_set_1"
                    else frac(n_deposits, total_deposits),
                    "top1_total_cells": total_cells,
                    "top1_total_deposits": total_deposits,
                }
            )

        total_fold_deposits = int(fold_df["target"].sum())
        zone0 = top1 & in_set & low
        zone1 = ((top1 & high) | (top1_5 & low)) & in_set
        zone01 = zone0 | zone1
        zone01_deposits = int(fold_df.loc[zone01, "target"].sum())
        r23_rows.append(
            {
                "model": model,
                "model_label": label,
                "scope": "fold",
                "fold": int(fold),
                "estimand": "R2.3_zone0_plus_zone1_cumulative_capture_pct",
                "n_cells": int(zone01.sum()),
                "n_deposits": zone01_deposits,
                "total_deposits": total_fold_deposits,
                "capture_pct": 100.0 * zone01_deposits / total_fold_deposits,
            }
        )

        top1_all = fold_df[top1]
        outside = int((top1_all["in_set_1"].astype(int) == 0).sum())
        diagnostic_rows.append(
            {
                "model": model,
                "model_label": label,
                "scope": "fold",
                "fold": int(fold),
                "top1_cells_without_in_set_filter": int(len(top1_all)),
                "top1_cells_outside_positive_conformal_set": outside,
                "top1_outside_positive_conformal_fraction": frac(
                    outside, int(len(top1_all))
                ),
                "corr_rho_iqr_bootstrap_mean_top1": corr_or_nan(top1_set),
            }
        )

    return split_rows, r23_rows, diagnostic_rows


def add_split_summaries(split_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for (model, group_name), group in split_df.groupby(
        ["model", "triage_group"], sort=True
    ):
        n_cells_mean = float(group["n_cells"].mean())
        n_deposits_mean = float(group["n_deposits"].mean())
        top_cells_mean = float(group["top1_total_cells"].mean())
        top_deposits_mean = float(group["top1_total_deposits"].mean())
        base = {
            "model": model,
            "model_label": group["model_label"].iloc[0],
            "tier": "top_1",
            "probability_col": "bootstrap_mean",
            "top1_rule": "per_fold_prob_ge_p99_and_in_set_1",
            "triage_group": group_name,
        }
        summary_rows.append(
            {
                **base,
                "scope": "fold_mean",
                "fold": "all",
                "n_cells": n_cells_mean,
                "n_cells_sd": float(group["n_cells"].std(ddof=1)),
                "n_deposits": n_deposits_mean,
                "n_deposits_sd": float(group["n_deposits"].std(ddof=1)),
                "cell_fraction_of_top1": frac(n_cells_mean, top_cells_mean),
                "deposit_fraction_of_top1": frac(n_deposits_mean, top_deposits_mean),
                "top1_total_cells": top_cells_mean,
                "top1_total_deposits": top_deposits_mean,
            }
        )
        n_cells = int(group["n_cells"].sum())
        n_deposits = int(group["n_deposits"].sum())
        top_cells = int(group["top1_total_cells"].sum())
        top_deposits = int(group["top1_total_deposits"].sum())
        summary_rows.append(
            {
                **base,
                "scope": "pooled",
                "fold": "all",
                "n_cells": n_cells,
                "n_cells_sd": math.nan,
                "n_deposits": n_deposits,
                "n_deposits_sd": math.nan,
                "cell_fraction_of_top1": frac(n_cells, top_cells),
                "deposit_fraction_of_top1": frac(n_deposits, top_deposits),
                "top1_total_cells": top_cells,
                "top1_total_deposits": top_deposits,
            }
        )
    return pd.concat([split_df, pd.DataFrame(summary_rows)], ignore_index=True)


def add_r23_summaries(r23_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in r23_df.groupby("model", sort=True):
        base = {
            "model": model,
            "model_label": group["model_label"].iloc[0],
            "fold": "all",
            "estimand": "R2.3_zone0_plus_zone1_cumulative_capture_pct",
        }
        rows.append(
            {
                **base,
                "scope": "fold_mean",
                "n_cells": float(group["n_cells"].mean()),
                "n_deposits": float(group["n_deposits"].mean()),
                "total_deposits": float(group["total_deposits"].mean()),
                "capture_pct": float(group["capture_pct"].mean()),
            }
        )
        n_deposits = int(group["n_deposits"].sum())
        total_deposits = int(group["total_deposits"].sum())
        rows.append(
            {
                **base,
                "scope": "pooled",
                "n_cells": int(group["n_cells"].sum()),
                "n_deposits": n_deposits,
                "total_deposits": total_deposits,
                "capture_pct": 100.0 * n_deposits / total_deposits,
            }
        )
    return pd.concat([r23_df, pd.DataFrame(rows)], ignore_index=True)


def add_diagnostic_summaries(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in diagnostic_df.groupby("model", sort=True):
        base = {
            "model": model,
            "model_label": group["model_label"].iloc[0],
            "fold": "all",
        }
        pooled_cells = int(group["top1_cells_without_in_set_filter"].sum())
        pooled_outside = int(group["top1_cells_outside_positive_conformal_set"].sum())
        rows.append(
            {
                **base,
                "scope": "fold_mean",
                "top1_cells_without_in_set_filter": float(
                    group["top1_cells_without_in_set_filter"].mean()
                ),
                "top1_cells_outside_positive_conformal_set": float(
                    group["top1_cells_outside_positive_conformal_set"].mean()
                ),
                "top1_outside_positive_conformal_fraction": float(
                    group["top1_outside_positive_conformal_fraction"].mean()
                ),
                "corr_rho_iqr_bootstrap_mean_top1": float(
                    group["corr_rho_iqr_bootstrap_mean_top1"].mean()
                ),
            }
        )
        rows.append(
            {
                **base,
                "scope": "pooled",
                "top1_cells_without_in_set_filter": pooled_cells,
                "top1_cells_outside_positive_conformal_set": pooled_outside,
                "top1_outside_positive_conformal_fraction": frac(
                    pooled_outside, pooled_cells
                ),
                "corr_rho_iqr_bootstrap_mean_top1": math.nan,
            }
        )
    return pd.concat([diagnostic_df, pd.DataFrame(rows)], ignore_index=True)


def gate_d_summary(split_df: pd.DataFrame) -> dict[str, Any]:
    expected = pd.read_csv(OUT / "D_rho_iqr_resolution_summary.csv")
    fold_mean = split_df[split_df["scope"].eq("fold_mean")]
    group_map = {
        "immediate_low_rho_le_0.30": "low_rho_le_0.30",
        "additional_evidence_high_rho_gt_0.30": "high_rho_gt_0.30",
    }
    checks = []
    for model in MODELS:
        for triage_group, d_group in group_map.items():
            actual = fold_mean[
                fold_mean["model"].eq(model)
                & fold_mean["triage_group"].eq(triage_group)
            ].iloc[0]
            target = expected[
                expected["model"].eq(model)
                & expected["tier"].eq("top_1")
                & expected["uncertainty_group"].eq(d_group)
            ].iloc[0]
            for metric, target_col in [
                ("n_cells", "n_cells_mean"),
                ("n_deposits", "n_deposits_mean"),
            ]:
                diff = abs(float(actual[metric]) - float(target[target_col]))
                checks.append(
                    {
                        "model": model,
                        "triage_group": triage_group,
                        "metric": metric,
                        "actual": float(actual[metric]),
                        "expected_from_D_summary": float(target[target_col]),
                        "abs_diff": diff,
                        "tolerance": D_SUMMARY_TOL,
                        "passed": diff <= D_SUMMARY_TOL,
                    }
                )
    return {
        "name": "D_summary top_1 high/low count reproduction",
        "passed": all(item["passed"] for item in checks),
        "details": {"checks": checks},
    }


def gate_r23_capture(r23_df: pd.DataFrame) -> dict[str, Any]:
    fold_mean = r23_df[r23_df["scope"].eq("fold_mean")]
    checks = []
    for model, expected in R23_CAPTURE_TARGET.items():
        actual = float(fold_mean[fold_mean["model"].eq(model)].iloc[0]["capture_pct"])
        diff = abs(actual - expected)
        checks.append(
            {
                "model": model,
                "actual_capture_pct": actual,
                "expected_approx_capture_pct": expected,
                "abs_diff_pp": diff,
                "tolerance_pp": R23_CAPTURE_TOL_PP,
                "passed": diff <= R23_CAPTURE_TOL_PP,
            }
        )
    return {
        "name": "R2.3 Immediate+Priority capture approx reproduction",
        "passed": all(item["passed"] for item in checks),
        "details": {"checks": checks},
    }


def gate_hashes(before: dict, after: dict) -> dict[str, Any]:
    checks = []
    for model in MODELS:
        checks.append(
            {
                "model": model,
                "path": before[model]["path"],
                "before_sha256": before[model]["sha256"],
                "after_sha256": after[model]["sha256"],
                "passed": before[model]["sha256"] == after[model]["sha256"],
            }
        )
    return {
        "name": "input CSV hashes unchanged",
        "passed": all(item["passed"] for item in checks),
        "details": {"checks": checks},
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def md_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(row[h] for h in headers) + " |" for row in rows]
    return "\n".join(lines)


def row_for(df: pd.DataFrame, model: str, scope: str, group: str) -> pd.Series:
    return df[
        df["model"].eq(model) & df["scope"].eq(scope) & df["triage_group"].eq(group)
    ].iloc[0]


def write_report(
    split_df: pd.DataFrame,
    r23_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    gates: list[dict[str, Any]],
) -> None:
    high = "additional_evidence_high_rho_gt_0.30"
    cite_rows = []
    pooled_rows = []
    distinct_rows = []
    diag_rows = []
    for model, label in MODELS.items():
        fm = row_for(split_df, model, "fold_mean", high)
        pooled = row_for(split_df, model, "pooled", high)
        r23_fm = r23_df[
            r23_df["model"].eq(model) & r23_df["scope"].eq("fold_mean")
        ].iloc[0]
        diag = diagnostic_df[
            diagnostic_df["model"].eq(model) & diagnostic_df["scope"].eq("fold_mean")
        ].iloc[0]
        cite_rows.append(
            {
                "model": label,
                "high-rho cells / top-1 cells": (
                    f"{float(fm['n_cells']):.1f}/{float(fm['top1_total_cells']):.1f}"
                ),
                "cell fraction": pct(float(fm["cell_fraction_of_top1"])),
                "high-rho deposits / top-1 deposits": (
                    f"{float(fm['n_deposits']):.1f}/"
                    f"{float(fm['top1_total_deposits']):.1f}"
                ),
                "deposit fraction": pct(float(fm["deposit_fraction_of_top1"])),
                "high-rho count sd": f"{float(fm['n_cells_sd']):.1f}",
                "high-rho deposit sd": f"{float(fm['n_deposits_sd']):.1f}",
            }
        )
        pooled_rows.append(
            {
                "model": label,
                "pooled high-rho cells / top-1 cells": (
                    f"{int(pooled['n_cells'])}/{int(pooled['top1_total_cells'])}"
                ),
                "pooled cell fraction": pct(float(pooled["cell_fraction_of_top1"])),
                "pooled high-rho deposits / top-1 deposits": (
                    f"{int(pooled['n_deposits'])}/{int(pooled['top1_total_deposits'])}"
                ),
                "pooled deposit fraction": pct(
                    float(pooled["deposit_fraction_of_top1"])
                ),
            }
        )
        distinct_rows.append(
            {
                "model": label,
                "R2.6 estimand": "high-rho fraction within top-1%",
                "R2.6 cell fraction": pct(float(fm["cell_fraction_of_top1"])),
                "R2.6 deposit fraction": pct(float(fm["deposit_fraction_of_top1"])),
                "R2.3 estimand": "Zone0+Zone1 capture among all deposits",
                "R2.3 capture": f"{float(r23_fm['capture_pct']):.2f}%",
            }
        )
        diag_rows.append(
            {
                "model": label,
                "top-1 outside in_set_1": pct(
                    float(diag["top1_outside_positive_conformal_fraction"])
                ),
                "corr(rho_IQR, bootstrap_mean), top-1": (
                    f"{float(diag['corr_rho_iqr_bootstrap_mean_top1']):.3f}"
                ),
            }
        )

    gate_status = "PASS" if all(gate["passed"] for gate in gates) else "FAIL"
    gate_rows = [
        {"gate": gate["name"], "status": "PASS" if gate["passed"] else "FAIL"}
        for gate in gates
    ]
    report = f"""# R2.6 triage numbers from frozen OOF predictions

## Methodology

- Objective: re-verify uncertainty-based triage of high-probability candidates for R2.6 without retraining.
- Frozen inputs: `outputs/aggregated_oof/{{xgboost,baggingpu_xgboost}}/merged_predictions.csv`.
- Probability surface: `bootstrap_mean`, matching the zone and R2.3 convention.
- Estimand: within each outer fold, compute the P99 threshold of `bootstrap_mean`, take cells with `bootstrap_mean >= P99`, keep the positive conformal set (`in_set_1 == 1`), and split the top-1% cells into `rho_IQR <= 30%` and `rho_IQR > 30%`.
- Primary summary: fold-mean over the five outer folds. Pooled values are secondary diagnostics.
- Interpretation: `rho_IQR <= 30%` is the immediate tier. `rho_IQR > 30%` is the additional-evidence tier. The latter remains a priority candidate group and is not treated as wrong or excluded.

## Numbers safe to cite in R2.6

Primary fold-mean R2.6 estimand: among top-1% `bootstrap_mean` candidates, the fraction assigned to the additional-evidence tier (`rho_IQR > 30%`).

{md_table(cite_rows)}

Secondary pooled diagnostic:

{md_table(pooled_rows)}

## Distinctness from R2.3

R2.6 and R2.3 use different denominators. R2.6 is a fraction of the top-1% candidate set. R2.3 Immediate+Priority is cumulative deposit capture by Zone0+Zone1, expressed as a percentage of all known deposits in the fold. These quantities should not be conflated.

{md_table(distinct_rows)}

## Diagnostics and caveat support

{md_table(diag_rows)}

The top-1% correlation diagnostic is narrower than the R2.3 upper-decile diagnostic. The R2.3 report gave upper-decile correlations of approximately -0.37 for standard XGBoost and -0.66 for BaggingPU-XGBoost, supporting the caveat that `rho_IQR` is partly shared with probability rather than a purely independent discovery signal.

## Hard gates

Overall: {gate_status}

{md_table(gate_rows)}

Detailed gate records are in `r26_hard_gates.json`.
"""
    (OUT / "r26_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = {
        model: {
            "path": str(input_path(model)),
            "sha256": sha256_file(input_path(model)),
        }
        for model in MODELS
    }

    split_rows: list[dict[str, Any]] = []
    r23_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for model, label in MODELS.items():
        split, r23, diagnostic = compute_model(model, label, load_oof(model))
        split_rows += split
        r23_rows += r23
        diagnostic_rows += diagnostic

    split_df = add_split_summaries(pd.DataFrame(split_rows))
    r23_df = add_r23_summaries(pd.DataFrame(r23_rows))
    diagnostic_df = add_diagnostic_summaries(pd.DataFrame(diagnostic_rows))
    split_df.to_csv(OUT / "r26_triage_split.csv", index=False)
    r23_df.to_csv(OUT / "r26_r23_distinctness.csv", index=False)
    diagnostic_df.to_csv(OUT / "r26_diagnostics.csv", index=False)

    after_hashes = {
        model: {
            "path": str(input_path(model)),
            "sha256": sha256_file(input_path(model)),
        }
        for model in MODELS
    }
    gates = [
        gate_d_summary(split_df),
        gate_r23_capture(r23_df),
        gate_hashes(before_hashes, after_hashes),
    ]
    for gate in gates:
        gate["status"] = "PASS" if gate["passed"] else "FAIL"
    payload = {
        "overall_status": "PASS" if all(gate["passed"] for gate in gates) else "FAIL",
        "script": str(Path(__file__).relative_to(MPM_ROOT.parent)),
        "inputs": {"before": before_hashes, "after": after_hashes},
        "parameters": {
            "probability_col": "bootstrap_mean",
            "rho_cut": RHO_CUT,
            "top1_rule": "per_fold_prob_ge_p99_and_in_set_1",
            "primary_summary": "fold_mean_over_outer_folds",
        },
        "gates": gates,
    }
    (OUT / "r26_hard_gates.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(split_df, r23_df, diagnostic_df, gates)

    for path in [
        OUT / "r26_triage_split.csv",
        OUT / "r26_r23_distinctness.csv",
        OUT / "r26_diagnostics.csv",
        OUT / "r26_hard_gates.json",
        OUT / "r26_report.md",
    ]:
        print(f"- {path.relative_to(MPM_ROOT.parent)}")
    print(f"Hard gates: {payload['overall_status']}")
    if payload["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
