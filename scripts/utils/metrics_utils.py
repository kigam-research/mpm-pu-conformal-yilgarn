#!/usr/bin/env python3
"""Metrics Utilities"""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall AUC (Average Precision)."""
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present, returning 0.0 for PR-AUC")
        return 0.0

    return average_precision_score(y_true, y_prob)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC AUC."""
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present, returning 0.5 for ROC-AUC")
        return 0.5

    return roc_auc_score(y_true, y_prob)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            'pr_auc': 0.0,
            'roc_auc': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'n_positives': int(y_true.sum()),
            'n_samples': len(y_true)
        }

    metrics = {
        'pr_auc': float(compute_pr_auc(y_true, y_prob)),
        'roc_auc': float(compute_roc_auc(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'n_positives': int(y_true.sum()),
        'n_samples': len(y_true)
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)

    return metrics


def compute_recall_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.1
) -> Dict:
    """Compute recall at a fixed false positive rate."""
    if len(np.unique(y_true)) < 2:
        return {'recall_at_fpr': 0.0, 'threshold': 0.5, 'actual_fpr': 0.0}

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    idx = np.argmin(np.abs(fpr - target_fpr))

    return {
        'recall_at_fpr': float(tpr[idx]),
        'threshold': float(thresholds[idx]) if idx < len(thresholds) else 0.5,
        'actual_fpr': float(fpr[idx]),
        'target_fpr': target_fpr
    }


def compute_deposit_capture_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 100
) -> Dict:
    """Compute deposit capture curve (cumulative capture vs area)."""
    total_deposits = y_true.sum()
    total_samples = len(y_true)

    if total_deposits == 0:
        return {
            'area_pct': [0, 100],
            'capture_pct': [0, 0],
            'auc': 0.0
        }

    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]

    area_pcts = np.linspace(0, 100, n_points)
    capture_pcts = []

    for area_pct in area_pcts:
        n_samples_top = int(len(y_true) * area_pct / 100)
        if n_samples_top == 0:
            capture_pcts.append(0.0)
        else:
            n_deposits_captured = y_sorted[:n_samples_top].sum()
            capture_pcts.append(float(n_deposits_captured / total_deposits * 100))

    auc = np.trapz(capture_pcts, area_pcts) / 10000

    return {
        'area_pct': area_pcts.tolist(),
        'capture_pct': capture_pcts,
        'auc': float(auc)
    }


def compute_top_k_capture(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_pcts: list = [1, 5, 10, 20]
) -> Dict:
    """Compute deposit capture at top k% of predictions."""
    total_deposits = y_true.sum()

    if total_deposits == 0:
        return {f'top_{k}pct': 0.0 for k in k_pcts}

    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]

    results = {}
    for k in k_pcts:
        n_top = int(len(y_true) * k / 100)
        if n_top == 0:
            n_top = 1
        captured = y_sorted[:n_top].sum()
        results[f'top_{k}pct'] = float(captured / total_deposits * 100)

    return results


def aggregate_fold_metrics(fold_metrics: list) -> Dict:
    """Aggregate metrics across folds."""
    if not fold_metrics:
        return {}

    all_keys = set()
    for m in fold_metrics:
        all_keys.update(m.keys())

    aggregated = {}

    for key in all_keys:
        values = [m.get(key) for m in fold_metrics if m.get(key) is not None]

        if not values:
            continue

        if not isinstance(values[0], (int, float)):
            continue

        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
        aggregated[f'{key}_min'] = float(np.min(values))
        aggregated[f'{key}_max'] = float(np.max(values))

    return aggregated
