"""Preprocessing Utilities"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)


class LeakageFreePreprocessor:
    """Preprocessing pipeline that prevents data leakage by fitting only on training data."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize preprocessor."""
        if config is None:
            from scripts.utils import load_config
            config = load_config()

        self.config = config

        self.proximity_features = config['data'].get('proximity_features', [])
        self.winsorize_features = config['data'].get('winsorize_features', [])

        self.impute_values_: Dict[str, float] = {}
        self.winsorize_bounds_: Dict[str, tuple] = {}
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: List[str] = []

        self.is_fitted_ = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> 'LeakageFreePreprocessor':
        """Fit preprocessor on training data."""
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            feature_names = list(X_df.columns)

        self.feature_names_ = feature_names
        logger.info(f"Fitting preprocessor on {len(X_df):,} samples, {len(feature_names)} features")

        logger.info("  Computing imputation values...")
        for col in feature_names:
            self.impute_values_[col] = X_df[col].mean()

        logger.info("  Computing winsorization bounds...")
        for col in feature_names:
            if col in self.winsorize_features:
                p1 = X_df[col].quantile(0.01)
                p99 = X_df[col].quantile(0.99)
                self.winsorize_bounds_[col] = (p1, p99)

        X_processed = self._apply_impute_winsorize_log(X_df)

        logger.info("  Fitting RobustScaler...")
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X_processed)

        self.is_fitted_ = True
        logger.info("  Preprocessor fitting complete")

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = self.feature_names_
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()

        X_processed = self._apply_impute_winsorize_log(X_df)

        X_scaled = self.scaler_.transform(X_processed)

        return X_scaled

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, feature_names)
        return self.transform(X, feature_names)

    def _apply_impute_winsorize_log(
        self,
        X_df: pd.DataFrame
    ) -> np.ndarray:
        """Apply imputation, winsorization, and log transform."""
        X_processed = X_df.copy()

        for col in self.feature_names_:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].fillna(self.impute_values_[col])

        for col, (p1, p99) in self.winsorize_bounds_.items():
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].clip(lower=p1, upper=p99)

        for col in self.proximity_features:
            if col in X_processed.columns:
                X_processed[col] = np.log1p(X_processed[col])

        return X_processed.values

    def get_params(self) -> dict:
        """Get fitted parameters for serialization."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted.")

        return {
            'feature_names': self.feature_names_,
            'impute_values': self.impute_values_,
            'winsorize_bounds': self.winsorize_bounds_,
            'scaler_center': self.scaler_.center_.tolist(),
            'scaler_scale': self.scaler_.scale_.tolist(),
            'proximity_features': self.proximity_features,
            'winsorize_features': self.winsorize_features
        }

    def set_params(self, params: dict) -> 'LeakageFreePreprocessor':
        """Set parameters from serialized dict."""
        self.feature_names_ = params['feature_names']
        self.impute_values_ = params['impute_values']
        self.winsorize_bounds_ = params['winsorize_bounds']
        self.proximity_features = params['proximity_features']
        self.winsorize_features = params['winsorize_features']

        self.scaler_ = RobustScaler()
        self.scaler_.center_ = np.array(params['scaler_center'])
        self.scaler_.scale_ = np.array(params['scaler_scale'])

        self.is_fitted_ = True
        return self
