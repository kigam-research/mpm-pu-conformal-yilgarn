#!/usr/bin/env python3
"""Conformal Prediction Utilities"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt calibration using logistic regression."""

    def __init__(self):
        self.calibrator_: Optional[LogisticRegression] = None
        self.coef_: float = 0.0
        self.intercept_: float = 0.0
        self.is_fitted_ = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'PlattCalibrator':
        """Fit Platt calibrator."""
        X_calib = y_prob.reshape(-1, 1)

        self.calibrator_ = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        self.calibrator_.fit(X_calib, y_true)

        self.coef_ = float(self.calibrator_.coef_[0, 0])
        self.intercept_ = float(self.calibrator_.intercept_[0])
        self.is_fitted_ = True

        logger.debug(f"Platt calibrator: coef={self.coef_:.4f}, intercept={self.intercept_:.4f}")
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt calibration."""
        if not self.is_fitted_:
            raise RuntimeError("Calibrator not fitted.")

        X = y_prob.reshape(-1, 1)
        return self.calibrator_.predict_proba(X)[:, 1]

    def get_params(self) -> Dict:
        """Get calibrator parameters."""
        return {
            'coef': self.coef_,
            'intercept': self.intercept_
        }


class PlattApplicator:
    """Apply pre-computed Platt calibration parameters."""

    def __init__(self, coef: float, intercept: float):
        """Initialize with pre-computed parameters."""
        self.coef_ = coef
        self.intercept_ = intercept

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration using stored parameters."""
        logits = self.coef_ * y_prob + self.intercept_
        return 1.0 / (1.0 + np.exp(-logits))


def fnr_control_direct(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    alpha_fnr: float = 0.15,
    delta: float = 0.05
) -> Tuple[float, Dict]:
    """FNR Control Direct algorithm (Angelopoulos et al. 2022)."""
    pos_mask = (y_true == 1)
    n_pos = pos_mask.sum()

    if n_pos == 0:
        logger.warning("No positive samples for FNR control!")
        return 0.0, {'fnr': 0.0, 'fnr_upper': 0.0, 'n_pos': 0}

    pos_probs = y_prob[pos_mask]

    hoeffding = np.sqrt(np.log(2.0 / delta) / (2.0 * n_pos))

    thresholds = np.sort(np.unique(y_prob))[::-1]

    for thresh in thresholds:
        fnr = 1.0 - (pos_probs >= thresh).mean()
        fnr_upper = fnr + hoeffding

        if fnr_upper <= alpha_fnr:
            logger.debug(
                f"FNR control: threshold={thresh:.4f}, "
                f"FNR={fnr:.4f}, FNR_upper={fnr_upper:.4f}"
            )
            return thresh, {
                'fnr': float(fnr),
                'fnr_upper': float(fnr_upper),
                'hoeffding': float(hoeffding),
                'n_pos': int(n_pos)
            }

    logger.warning("No threshold satisfies FNR constraint, using minimum")
    return float(thresholds[-1]), {
        'fnr': 1.0,
        'fnr_upper': 1.0 + hoeffding,
        'hoeffding': float(hoeffding),
        'n_pos': int(n_pos)
    }


def compute_prediction_sets(
    calibrated_prob: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Compute prediction sets (which classes are included)."""
    in_set_1 = calibrated_prob >= threshold
    in_set_0 = (1.0 - calibrated_prob) >= threshold

    return np.column_stack([in_set_0, in_set_1])


def compute_fnr_practical_zones(
    calibrated_prob: np.ndarray,
    bootstrap_std: np.ndarray,
    in_set_0: np.ndarray,
    in_set_1: np.ndarray,
    zone_thresholds: Dict[str, float],
    base_rate: Optional[float] = None
) -> np.ndarray:
    """Compute FNR-based practical zones using Percentile + Conformal + Uncertainty."""
    n_samples = len(calibrated_prob)
    zones = np.full(n_samples, 4, dtype=int)

    use_percentile = zone_thresholds.get('use_percentile_zones', True)

    if use_percentile:
        prob_pct_high = zone_thresholds.get('prob_percentile_high', 98)
        prob_pct_mod = zone_thresholds.get('prob_percentile_mod', 90)
        prob_pct_low = zone_thresholds.get('prob_percentile_low', 70)

        prob_threshold_high = np.percentile(calibrated_prob, prob_pct_high)
        prob_threshold_mod = np.percentile(calibrated_prob, prob_pct_mod)
        prob_threshold_low = np.percentile(calibrated_prob, prob_pct_low)

        std_low = zone_thresholds.get('std_low', 0.0765)
        std_moderate = zone_thresholds.get('std_moderate', 0.1020)
        std_high = zone_thresholds.get('std_high', 0.1276)

        logger.info(f"Percentile-based zone thresholds:")
        logger.info(f"  Prob: P{prob_pct_high}={prob_threshold_high:.4f}, "
                   f"P{prob_pct_mod}={prob_threshold_mod:.4f}, "
                   f"P{prob_pct_low}={prob_threshold_low:.4f}")
        logger.info(f"  Std (95% CI width): low={std_low:.4f} (<30%), "
                   f"mod={std_moderate:.4f} (<40%), high={std_high:.4f} (>50%)")

        for i in range(n_samples):
            prob = calibrated_prob[i]
            std = bootstrap_std[i]
            has_class_1 = in_set_1[i]

            if has_class_1 and prob >= prob_threshold_high and std < std_low:
                zones[i] = 0
            elif has_class_1 and prob >= prob_threshold_mod and std < std_moderate:
                zones[i] = 1
            elif has_class_1 and prob >= prob_threshold_low:
                zones[i] = 2
            elif has_class_1:
                zones[i] = 3
            elif std > std_high:
                zones[i] = 5
            else:
                zones[i] = 4
    else:
        if base_rate is None:
            base_rate = 0.01

        mult_high = zone_thresholds.get('base_rate_mult_high', 5.0)
        mult_moderate = zone_thresholds.get('base_rate_mult_moderate', 2.0)
        mult_low = zone_thresholds.get('base_rate_mult_low', 1.5)

        prob_threshold_high = base_rate * mult_high
        prob_threshold_mod = base_rate * mult_moderate
        prob_threshold_low = base_rate * mult_low

        std_low = zone_thresholds.get('std_low', 0.0765)
        std_moderate = zone_thresholds.get('std_moderate', 0.1020)
        std_high = zone_thresholds.get('std_high', 0.1276)

        logger.info(f"Legacy zone thresholds (base_rate={base_rate:.4f}):")
        logger.info(f"  Prob: {mult_high}x={prob_threshold_high:.4f}, "
                   f"{mult_moderate}x={prob_threshold_mod:.4f}, "
                   f"{mult_low}x={prob_threshold_low:.4f}")

        for i in range(n_samples):
            prob = calibrated_prob[i]
            std = bootstrap_std[i]
            has_class_1 = in_set_1[i]

            if has_class_1 and prob >= prob_threshold_high and std < std_low:
                zones[i] = 0
            elif has_class_1 and prob >= prob_threshold_mod and std < std_moderate:
                zones[i] = 1
            elif has_class_1 and prob >= prob_threshold_low:
                zones[i] = 2
            elif has_class_1:
                zones[i] = 3
            elif std > std_high:
                zones[i] = 5
            else:
                zones[i] = 4

    for z in range(6):
        count = (zones == z).sum()
        pct = count / n_samples * 100
        logger.debug(f"Zone {z}: {count:,} samples ({pct:.1f}%)")

    return zones


def assign_zones_jorc30(
    df: 'pd.DataFrame',
    prob_col: str = 'bootstrap_mean',
    q25_col: str = 'q25',
    q75_col: str = 'q75',
    in_set_1_col: str = 'in_set_1',
    rel_iqr_threshold: float = 0.30,
    prob_z0: float = 0.50,
    prob_z1: float = 0.25,
    prob_z2: float = 0.10
) -> np.ndarray:
    """Assign zones under the JORC 30% criterion (5 zones: 0-4)."""
    prob = df[prob_col].values
    q25 = df[q25_col].values
    q75 = df[q75_col].values
    iqr = q75 - q25
    epsilon = 1e-10
    rel_iqr = iqr / (prob + epsilon)
    in_set_1 = df[in_set_1_col].values.astype(bool)

    n = len(df)
    zones = np.full(n, -1, dtype=int)

    zones[~in_set_1] = 4

    z0_mask = (prob >= prob_z0) & (rel_iqr <= rel_iqr_threshold) & in_set_1
    zones[z0_mask] = 0

    z1_cond1 = (prob >= prob_z0) & (rel_iqr > rel_iqr_threshold) & in_set_1
    z1_cond2 = (prob >= prob_z1) & (prob < prob_z0) & (rel_iqr <= rel_iqr_threshold) & in_set_1
    zones[z1_cond1 | z1_cond2] = 1

    z2_cond1 = (prob >= prob_z1) & (prob < prob_z0) & (rel_iqr > rel_iqr_threshold) & in_set_1
    z2_cond2 = (prob >= prob_z2) & (prob < prob_z1) & (rel_iqr <= rel_iqr_threshold) & in_set_1
    zones[z2_cond1 | z2_cond2] = 2

    z3_cond1 = (prob >= prob_z2) & (prob < prob_z1) & (rel_iqr > rel_iqr_threshold) & in_set_1
    z3_cond2 = (prob < prob_z2) & in_set_1
    zones[z3_cond1 | z3_cond2] = 3

    unassigned = (zones == -1).sum()
    if unassigned > 0:
        logger.warning(f"Unassigned samples: {unassigned}")

    for z in range(5):
        count = (zones == z).sum()
        pct = count / n * 100
        logger.debug(f"Zone {z}: {count:,} samples ({pct:.1f}%)")

    return zones


def assign_zones_lift(
    df: 'pd.DataFrame',
    prob_col: str = 'bootstrap_mean',
    q25_col: str = 'q25',
    q75_col: str = 'q75',
    in_set_1_col: str = 'in_set_1',
    rel_iqr_threshold: float = 0.30,
    lift_z0: float = 0.01,
    lift_z1: float = 0.05,
    lift_z2: float = 0.10
) -> np.ndarray:
    """Assign zones using model-specific lift percentiles (5 zones: 0-4)."""
    prob = df[prob_col].values
    q25 = df[q25_col].values
    q75 = df[q75_col].values
    iqr = q75 - q25
    epsilon = 1e-10
    rel_iqr = iqr / (prob + epsilon)
    in_set_1 = df[in_set_1_col].values.astype(bool)

    p_z0 = np.percentile(prob, 100 - lift_z0 * 100)
    p_z1 = np.percentile(prob, 100 - lift_z1 * 100)
    p_z2 = np.percentile(prob, 100 - lift_z2 * 100)

    logger.info(f"Lift-based percentile thresholds:")
    logger.info(f"  Top {lift_z0*100:.0f}%: prob >= {p_z0:.4f}")
    logger.info(f"  Top {lift_z1*100:.0f}%: prob >= {p_z1:.4f}")
    logger.info(f"  Top {lift_z2*100:.0f}%: prob >= {p_z2:.4f}")

    n = len(df)
    zones = np.full(n, -1, dtype=int)

    zones[~in_set_1] = 4

    top_z0 = prob >= p_z0
    top_z1 = (prob >= p_z1) & (prob < p_z0)
    top_z2 = (prob >= p_z2) & (prob < p_z1)
    below_z2 = prob < p_z2

    z0_mask = top_z0 & (rel_iqr <= rel_iqr_threshold) & in_set_1
    zones[z0_mask] = 0

    z1_cond1 = top_z0 & (rel_iqr > rel_iqr_threshold) & in_set_1
    z1_cond2 = top_z1 & (rel_iqr <= rel_iqr_threshold) & in_set_1
    zones[z1_cond1 | z1_cond2] = 1

    z2_cond1 = top_z1 & (rel_iqr > rel_iqr_threshold) & in_set_1
    z2_cond2 = top_z2 & (rel_iqr <= rel_iqr_threshold) & in_set_1
    zones[z2_cond1 | z2_cond2] = 2

    z3_cond1 = top_z2 & (rel_iqr > rel_iqr_threshold) & in_set_1
    z3_cond2 = below_z2 & in_set_1
    zones[z3_cond1 | z3_cond2] = 3

    unassigned = (zones == -1).sum()
    if unassigned > 0:
        logger.warning(f"Unassigned samples: {unassigned}")

    for z in range(5):
        count = (zones == z).sum()
        pct = count / n * 100
        logger.debug(f"Zone {z}: {count:,} samples ({pct:.1f}%)")

    return zones


def compute_coverage_metrics(
    y_true: np.ndarray,
    in_set_1: np.ndarray
) -> Dict:
    """Compute coverage metrics."""
    pos_mask = y_true == 1
    n_pos = pos_mask.sum()

    if n_pos == 0:
        return {
            'class_1_coverage': 0.0,
            'fnr': 1.0,
            'n_positives': 0
        }

    coverage = in_set_1[pos_mask].mean()
    fnr = 1.0 - coverage

    return {
        'class_1_coverage': float(coverage),
        'fnr': float(fnr),
        'n_positives': int(n_pos)
    }


def compute_deposit_capture_by_zone(
    zones: np.ndarray,
    y_true: np.ndarray,
    n_zones: Optional[int] = None
) -> Dict:
    """Compute deposit capture statistics by zone."""
    total_deposits = y_true.sum()
    total_samples = len(y_true)

    if n_zones is None:
        n_zones = int(zones.max()) + 1

    zone_stats = {}
    cumulative_deposits = 0

    for z in range(n_zones):
        zone_mask = zones == z
        n_samples = zone_mask.sum()
        n_deposits = y_true[zone_mask].sum()

        cumulative_deposits += n_deposits

        zone_stats[f'zone_{z}'] = {
            'n_samples': int(n_samples),
            'n_deposits': int(n_deposits),
            'sample_pct': float(n_samples / total_samples * 100),
            'deposit_pct': float(n_deposits / total_deposits * 100) if total_deposits > 0 else 0,
            'cumulative_deposit_pct': float(cumulative_deposits / total_deposits * 100) if total_deposits > 0 else 0
        }

    zone_stats['total'] = {
        'n_samples': int(total_samples),
        'n_deposits': int(total_deposits)
    }
    zone_stats['n_zones'] = n_zones

    return zone_stats
