"""Leakage-safe feature engineering and scikit-learn preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config.settings import (
    ALL_FEATURE_COLUMNS,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("feature_pipeline")


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer to engineer domain-specific property features without target leakage."""

    def __init__(self):
        self.engineered_cols = ENGINEERED_NUMERICAL_FEATURES

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=ALL_FEATURE_COLUMNS)

        # 1. Area to BHK ratio
        bhk_safe = np.maximum(pd.to_numeric(X_df["bhk"], errors="coerce").fillna(2), 1.0)
        area_safe = pd.to_numeric(X_df["area_sqft"], errors="coerce").fillna(1000.0)
        X_df["area_per_bhk"] = area_safe / bhk_safe

        # 2. Floor ratio relative to building height
        floor_num = pd.to_numeric(X_df.get("floor_number", 2), errors="coerce").fillna(2)
        total_fl = pd.to_numeric(X_df.get("total_floors", 4), errors="coerce").fillna(4)
        total_fl_safe = np.maximum(total_fl, 1.0)
        X_df["floor_ratio"] = (floor_num / total_fl_safe).clip(lower=0.0, upper=1.0)

        # 3. Bath to BHK ratio
        bath_safe = pd.to_numeric(X_df.get("bathrooms", 2), errors="coerce").fillna(2)
        X_df["bath_to_bhk_ratio"] = (bath_safe / bhk_safe).clip(lower=0.2, upper=4.0)

        return X_df


def create_preprocessor(min_locality_freq: int = 5) -> ColumnTransformer:
    """Create a composite ColumnTransformer for numeric, categorical, and binary inputs."""
    all_numeric = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=min_locality_freq,
                ),
            ),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_full_pipeline(
    regressor,
    min_locality_freq: int = 5,
    use_log_target: bool = True,
) -> Pipeline:
    """Construct an end-to-end pipeline with feature engineering, preprocessing, and regressor.

    If use_log_target is True, the regressor is wrapped in a TransformedTargetRegressor
    with log1p and expm1 transformations to guarantee non-negative predictions.
    """
    preprocessor = create_preprocessor(min_locality_freq=min_locality_freq)

    core_pipeline = Pipeline(
        steps=[
            ("feature_engineer", HousingFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    if use_log_target:
        # Wrap final regressor in TransformedTargetRegressor with log1p/expm1
        wrapped_model = TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        return Pipeline(
            steps=[
                ("feature_engineer", HousingFeatureEngineer()),
                ("preprocessor", preprocessor),
                ("model", wrapped_model),
            ]
        )

    return core_pipeline
