"""Tests for feature engineering, preprocessors, and target leakage safety."""

import numpy as np

from src.config.settings import ALL_FEATURE_COLUMNS, TARGET_COLUMN
from src.features.feature_pipeline import (
    HousingFeatureEngineer,
    create_preprocessor,
)


def test_feature_engineer_adds_expected_columns(sample_valid_dataframe):
    fe = HousingFeatureEngineer()
    X = sample_valid_dataframe[ALL_FEATURE_COLUMNS].copy()
    transformed = fe.transform(X)

    assert "area_per_bhk" in transformed.columns
    assert "floor_ratio" in transformed.columns
    assert "bath_to_bhk_ratio" in transformed.columns

    # Verify calculation logic
    expected_area_per_bhk = transformed["area_sqft"].iloc[0] / transformed["bhk"].iloc[0]
    assert np.isclose(transformed["area_per_bhk"].iloc[0], expected_area_per_bhk)


def test_no_target_leakage_in_features(sample_valid_dataframe):
    """Ensure price_inr and price-per-sqft are never in the feature set."""
    assert TARGET_COLUMN not in ALL_FEATURE_COLUMNS
    for col in ALL_FEATURE_COLUMNS:
        assert "price" not in col.lower()
        assert "target" not in col.lower()


def test_preprocessor_handles_unknown_categories(sample_valid_dataframe):
    X = sample_valid_dataframe[ALL_FEATURE_COLUMNS].copy()
    fe = HousingFeatureEngineer()
    X_fe = fe.transform(X)

    preprocessor = create_preprocessor(min_locality_freq=1)
    preprocessor.fit(X_fe)

    # Create unseen category input
    unseen_row = X_fe.iloc[[0]].copy()
    unseen_row["locality"] = "CompletelyUnknownLocality123"
    unseen_row["property_type"] = "UnknownType456"

    # Preprocessing must transform without throwing exception
    transformed = preprocessor.transform(unseen_row)
    assert transformed is not None
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
