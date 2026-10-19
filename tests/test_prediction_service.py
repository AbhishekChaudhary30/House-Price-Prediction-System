"""Tests for prediction service, input validation, and location coverage handling."""

import pytest
from sklearn.linear_model import Ridge

from src.config.settings import ALL_FEATURE_COLUMNS, TARGET_COLUMN
from src.features.feature_pipeline import build_full_pipeline
from src.services.prediction_service import (
    HousingPredictionService,
    PredictionRequest,
)


@pytest.fixture
def test_service(sample_valid_dataframe) -> HousingPredictionService:
    """Instantiate a lightweight PredictionService using an in-memory trained Ridge pipeline."""
    X = sample_valid_dataframe[ALL_FEATURE_COLUMNS]
    y = sample_valid_dataframe[TARGET_COLUMN]

    pipeline = build_full_pipeline(Ridge(alpha=1.0), min_locality_freq=1, use_log_target=True)
    pipeline.fit(X, y)

    registry = {
        "cities": {
            "Bengaluru": {
                "observation_count": 10,
                "localities": {
                    "Whitefield": {"observation_count": 15, "is_supported": True},
                },
            },
            "Mumbai": {
                "observation_count": 10,
                "localities": {
                    "Andheri West": {"observation_count": 12, "is_supported": True},
                },
            },
            "Delhi": {
                "observation_count": 10,
                "localities": {
                    "Rohini": {"observation_count": 10, "is_supported": True},
                },
            },
        }
    }

    service = HousingPredictionService(pipeline=pipeline, metadata={"version": "test_1.0"})
    service.location_registry = registry
    return service


def test_valid_request_returns_success(test_service):
    req = PredictionRequest(
        city="Bengaluru",
        locality="Whitefield",
        area_value=1200.0,
        area_unit="sqft",
        bhk=2,
        bathrooms=2,
        floor_number=2,
        total_floors=4,
    )
    resp = test_service.predict(req)
    assert resp.success is True
    assert resp.estimated_price_inr > 0
    assert "₹" in resp.formatted_price
    assert resp.location_support_status == "SUPPORTED_LOCALITY"
    assert resp.confidence_tier == "HIGH"


def test_city_alias_normalization(test_service):
    req = PredictionRequest(
        city="bangalore",  # Lowercase alias
        locality="Whitefield",
        area_value=1200.0,
        bhk=2,
        bathrooms=2,
        floor_number=2,
        total_floors=4,
    )
    resp = test_service.predict(req)
    assert resp.success is True
    assert resp.city == "Bengaluru"


def test_unsupported_city_rejected_without_substitution(test_service):
    req = PredictionRequest(
        city="Patna",  # Unsupported city
        locality="Boring Road",
        area_value=1000.0,
        bhk=2,
        bathrooms=2,
    )
    resp = test_service.predict(req)
    assert resp.success is False
    assert resp.location_support_status == "UNSUPPORTED_CITY"
    assert any("not represented in the verified training dataset" in lim for lim in resp.limitations)
    # Ensure it did NOT silently substitute
    assert resp.city == "Patna"


def test_invalid_floor_number_exceeding_total_floors(test_service):
    req = PredictionRequest(
        city="Bengaluru",
        locality="Whitefield",
        area_value=1200.0,
        bhk=2,
        bathrooms=2,
        floor_number=15,  # Floor 15 in a 4-floor building
        total_floors=4,
    )
    resp = test_service.predict(req)
    assert resp.success is False
    assert any("cannot exceed total building floors" in err for err in resp.validation_errors)


def test_invalid_negative_or_extreme_area(test_service):
    req = PredictionRequest(
        city="Bengaluru",
        locality="Whitefield",
        area_value=25.0,  # 25 sqft is implausibly tiny
        bhk=2,
        bathrooms=2,
    )
    resp = test_service.predict(req)
    assert resp.success is False
    assert any("falls outside plausible residential range" in err for err in resp.validation_errors)
