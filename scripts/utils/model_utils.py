#!/usr/bin/env python3
"""Model Utilities"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class BaggingPUClassifier:
    """BaggingPU Classifier for Positive-Unlabeled learning."""

    def __init__(
        self,
        base_estimator_params: dict,
        n_estimators: int = 15,
        max_samples: float = 0.35,
        random_state: int = 42
    ):
        """Initialize BaggingPU classifier."""
        self.base_estimator_params = base_estimator_params
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators_: List[XGBClassifier] = []
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingPUClassifier':
        """Fit BaggingPU classifier."""
        np.random.seed(self.random_state)

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        n_neg_sample = int(n_neg * self.max_samples)

        logger.info(
            f"BaggingPU: {n_pos} positives, sampling {n_neg_sample}/{n_neg} negatives per bag"
        )

        self.estimators_ = []

        for i in range(self.n_estimators):
            sampled_neg_idx = np.random.choice(neg_idx, n_neg_sample, replace=False)

            bag_idx = np.concatenate([pos_idx, sampled_neg_idx])
            np.random.shuffle(bag_idx)

            X_bag = X[bag_idx]
            y_bag = y[bag_idx]

            estimator = XGBClassifier(**self.base_estimator_params)
            estimator.fit(X_bag, y_bag)
            self.estimators_.append(estimator)

            if (i + 1) % 5 == 0:
                logger.info(f"  Trained {i + 1}/{self.n_estimators} bags")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted.")

        proba_list = []
        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)
            proba_list.append(proba)

        mean_proba = np.mean(proba_list, axis=0)
        return mean_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_individual_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from each individual estimator."""
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted.")

        predictions = []
        for estimator in self.estimators_:
            proba = estimator.predict_proba(X)[:, 1]
            predictions.append(proba)

        return np.array(predictions)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict
) -> XGBClassifier:
    """Train XGBoost classifier."""
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_baggingpu_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bagging_params: dict,
    base_estimator_params: dict
) -> BaggingPUClassifier:
    """Train BaggingPU(XGBoost) classifier."""
    model = BaggingPUClassifier(
        base_estimator_params=base_estimator_params,
        **bagging_params
    )
    model.fit(X_train, y_train)
    return model


def train_bootstrap_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    model_params: dict,
    n_bootstraps: int = 50,
    random_state: int = 42,
    save_dir: Optional[Path] = None
) -> Tuple[List, np.ndarray, np.ndarray]:
    """Train bootstrap ensemble for uncertainty quantification."""
    np.random.seed(random_state)
    n_samples = len(X_train)

    models = []
    bootstrap_predictions = []

    for i in range(n_bootstraps):
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]

        if model_type == 'xgboost':
            model = train_xgboost(X_boot, y_boot, model_params)
        elif model_type == 'baggingpu_xgboost':
            bagging_params = {
                'n_estimators': model_params.pop('bagging_n_estimators', 15),
                'max_samples': model_params.pop('bagging_max_samples', 0.35),
                'random_state': random_state + i
            }
            model = train_baggingpu_xgboost(X_boot, y_boot, bagging_params, model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        models.append(model)

        if save_dir is not None:
            model_path = save_dir / f"bootstrap_{i:03d}.joblib"
            joblib.dump(model, model_path)

        if (i + 1) % 10 == 0:
            logger.info(f"Trained bootstrap {i + 1}/{n_bootstraps}")
            gc.collect()

    return models


def predict_bootstrap_ensemble(
    models: List,
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from bootstrap ensemble."""
    predictions = []

    for model in models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X)[:, 1]
        else:
            pred = model.predict(X)
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred


def load_bootstrap_models(model_dir: Path) -> List:
    """Load bootstrap models from directory."""
    model_files = sorted(model_dir.glob("bootstrap_*.joblib"))
    models = [joblib.load(f) for f in model_files]
    logger.info(f"Loaded {len(models)} bootstrap models from {model_dir}")
    return models


def save_model(model, path: Path) -> None:
    """Save model to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model to: {path}")


def load_model(path: Path):
    """Load model from file."""
    return joblib.load(path)
