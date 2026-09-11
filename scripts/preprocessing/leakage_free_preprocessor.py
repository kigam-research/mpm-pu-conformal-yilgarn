#!/usr/bin/env python3
"""03: Leakage-Free Preprocessor"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler
from typing import List, Dict, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml"""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class LeakageFreePreprocessor:
    """Preprocessing pipeline that prevents data leakage by fitting only on training data."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize preprocessor."""
        if config is None:
            config = load_config()

        self.config = config

        self.proximity_features = config['data'].get('proximity_features', [])
        self.winsorize_features = config['data'].get('winsorize_features', [])
        self.missing_indicator_threshold = config['data'].get('missing_indicator_threshold', 0.05)

        self.impute_values_: Dict[str, float] = {}
        self.winsorize_bounds_: Dict[str, tuple] = {}
        self.missing_indicator_cols_: List[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: List[str] = []
        self.final_feature_names_: List[str] = []

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

        logger.info("  Checking missing rates...")
        self.missing_indicator_cols_ = []
        for col in feature_names:
            missing_rate = X_df[col].isna().mean()
            if missing_rate > self.missing_indicator_threshold:
                self.missing_indicator_cols_.append(col)
                logger.info(f"    {col}: {missing_rate*100:.2f}% missing -> adding indicator")

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

        self.final_feature_names_ = list(feature_names)
        for col in self.missing_indicator_cols_:
            self.final_feature_names_.append(f"{col}_missing")

        logger.info("  Fitting RobustScaler...")
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X_processed)

        self.is_fitted_ = True
        logger.info(f"  Preprocessor fitting complete ({len(self.final_feature_names_)} final features)")

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
        """Apply missing indicator, imputation, winsorization, and log transform."""
        X_processed = X_df.copy()

        missing_indicators = {}
        for col in self.missing_indicator_cols_:
            if col in X_processed.columns:
                missing_indicators[f"{col}_missing"] = X_processed[col].isna().astype(float)

        for col in self.feature_names_:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].fillna(self.impute_values_[col])

        for col, (p1, p99) in self.winsorize_bounds_.items():
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].clip(lower=p1, upper=p99)

        for col in self.proximity_features:
            if col in X_processed.columns:
                X_processed[col] = np.log1p(X_processed[col])

        for indicator_name, indicator_values in missing_indicators.items():
            X_processed[indicator_name] = indicator_values

        final_cols = list(self.feature_names_) + [f"{col}_missing" for col in self.missing_indicator_cols_]
        return X_processed[final_cols].values

    def get_params(self) -> dict:
        """Get fitted parameters for serialization."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted.")

        return {
            'feature_names': self.feature_names_,
            'final_feature_names': self.final_feature_names_,
            'impute_values': self.impute_values_,
            'winsorize_bounds': self.winsorize_bounds_,
            'missing_indicator_cols': self.missing_indicator_cols_,
            'scaler_center': self.scaler_.center_.tolist(),
            'scaler_scale': self.scaler_.scale_.tolist(),
            'proximity_features': self.proximity_features,
            'winsorize_features': self.winsorize_features,
            'missing_indicator_threshold': self.missing_indicator_threshold
        }

    def set_params(self, params: dict) -> 'LeakageFreePreprocessor':
        """Set parameters from serialized dict."""
        self.feature_names_ = params['feature_names']
        self.final_feature_names_ = params.get('final_feature_names', params['feature_names'])
        self.impute_values_ = params['impute_values']
        self.winsorize_bounds_ = params['winsorize_bounds']
        self.missing_indicator_cols_ = params.get('missing_indicator_cols', [])
        self.proximity_features = params['proximity_features']
        self.winsorize_features = params['winsorize_features']
        self.missing_indicator_threshold = params.get('missing_indicator_threshold', 0.05)

        self.scaler_ = RobustScaler()
        self.scaler_.center_ = np.array(params['scaler_center'])
        self.scaler_.scale_ = np.array(params['scaler_scale'])

        self.is_fitted_ = True
        return self


def main():
    """Test the preprocessor with sample data."""
    logger.info("=" * 70)
    logger.info("Testing LeakageFreePreprocessor")
    logger.info("=" * 70)

    config = load_config()

    data_path = config['paths']['original_data']
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {len(df):,} samples")

    feature_cols = config['data']['feature_columns']
    available_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_cols)} features")

    X = df[available_cols]

    n_train = int(len(X) * 0.8)
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]

    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    preprocessor = LeakageFreePreprocessor(config)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"\nProcessed train shape: {X_train_processed.shape}")
    logger.info(f"Processed test shape: {X_test_processed.shape}")

    train_nan = np.isnan(X_train_processed).sum()
    test_nan = np.isnan(X_test_processed).sum()
    logger.info(f"\nNaN count - Train: {train_nan}, Test: {test_nan}")

    logger.info(f"\nTrain mean: {X_train_processed.mean(axis=0)[:3]}")
    logger.info(f"Train std: {X_train_processed.std(axis=0)[:3]}")

    logger.info("\nPreprocessor test completed successfully!")


if __name__ == "__main__":
    main()
