"""Tests for dataset validation, schema enforcement, and data cleaning checks."""

import pandas as pd

from src.data.clean_data import filter_plausible_records
from src.data.validate_data import validate_dataset_schema


def test_valid_dataset_passes_validation(sample_valid_dataframe):
    report = validate_dataset_schema(sample_valid_dataframe, require_target=True)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.row_count == len(sample_valid_dataframe)


def test_missing_required_column_fails(sample_valid_dataframe):
    corrupted = sample_valid_dataframe.drop(columns=["area_sqft"])
    report = validate_dataset_schema(corrupted, require_target=True)
    assert report.is_valid is False
    assert any("area_sqft" in err for err in report.errors)


def test_missing_target_column_fails(sample_valid_dataframe):
    corrupted = sample_valid_dataframe.drop(columns=["price_inr"])
    report = validate_dataset_schema(corrupted, require_target=True)
    assert report.is_valid is False
    assert any("Missing required target column" in err for err in report.errors)


def test_negative_target_price_fails(sample_valid_dataframe):
    corrupted = sample_valid_dataframe.copy()
    corrupted.loc[0, "price_inr"] = -500000.0
    report = validate_dataset_schema(corrupted, require_target=True)
    assert report.is_valid is False
    assert any("non-positive" in err for err in report.errors)


def test_non_numeric_bhk_fails(sample_valid_dataframe):
    corrupted = sample_valid_dataframe.copy()
    corrupted["bhk"] = "invalid_string"
    report = validate_dataset_schema(corrupted, require_target=True)
    assert report.is_valid is False
    assert any("must be numeric" in err for err in report.errors)


def test_duplicate_records_detected_as_warning(sample_valid_dataframe):
    duplicated = pd.concat([sample_valid_dataframe, sample_valid_dataframe.iloc[[0]]], ignore_index=True)
    report = validate_dataset_schema(duplicated, require_target=True)
    assert report.is_valid is True  # Duplicates generate warnings, not schema failure
    assert any("duplicate" in w for w in report.warnings)


def test_filter_plausible_records_removes_extreme_errors(sample_valid_dataframe):
    df_with_errors = sample_valid_dataframe.copy()
    # Add an absurd price error: ₹50 for 1000 sqft
    df_with_errors.loc[len(df_with_errors)] = {
        "city": "Bengaluru",
        "locality": "Whitefield",
        "property_type": "Apartment",
        "bhk": 2,
        "bathrooms": 2,
        "area_sqft": 1000.0,
        "floor_number": 2,
        "total_floors": 4,
        "furnishing_status": "Semi-Furnished",
        "parking_available": 1,
        "lift_available": 1,
        "gated_community": 1,
        "price_inr": 50.0,  # Extreme error
    }
    filtered = filter_plausible_records(df_with_errors)
    assert len(filtered) == len(sample_valid_dataframe)
