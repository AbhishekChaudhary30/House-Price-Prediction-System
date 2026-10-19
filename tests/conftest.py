"""Pytest fixtures for unit and integration testing."""

import pandas as pd
import pytest


@pytest.fixture
def sample_valid_dataframe() -> pd.DataFrame:
    """Fixture providing a controlled small synthetic dataset strictly conforming to schema."""
    data = {
        "city": ["Bengaluru", "Bengaluru", "Mumbai", "Mumbai", "Delhi", "Delhi", "Bengaluru", "Mumbai"],
        "locality": ["Whitefield", "Whitefield", "Andheri West", "Andheri West", "Rohini", "Rohini", "Sarjapur Road", "Bandra West"],
        "property_type": ["Apartment", "Apartment", "Apartment", "Apartment", "Builder Floor", "Builder Floor", "Independent House", "Apartment"],
        "bhk": [2, 3, 2, 3, 2, 3, 4, 3],
        "bathrooms": [2, 3, 2, 3, 2, 2, 4, 3],
        "area_sqft": [1100.0, 1600.0, 850.0, 1250.0, 900.0, 1400.0, 2400.0, 1500.0],
        "floor_number": [2, 4, 5, 8, 2, 3, 1, 10],
        "total_floors": [4, 12, 14, 16, 4, 4, 2, 20],
        "furnishing_status": ["Semi-Furnished", "Furnished", "Unfurnished", "Semi-Furnished", "Semi-Furnished", "Furnished", "Furnished", "Furnished"],
        "parking_available": [1, 1, 1, 1, 1, 1, 1, 1],
        "lift_available": [1, 1, 1, 1, 0, 0, 0, 1],
        "gated_community": [1, 1, 1, 1, 0, 0, 1, 1],
        "price_inr": [6_500_000.0, 11_000_000.0, 18_000_000.0, 27_000_000.0, 5_500_000.0, 9_500_000.0, 22_000_000.0, 45_000_000.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_property_input_dict() -> dict:
    """Fixture providing a single valid property input dictionary."""
    return {
        "city": "Bengaluru",
        "locality": "Whitefield",
        "property_type": "Apartment",
        "bhk": 2,
        "bathrooms": 2,
        "area_sqft": 1200.0,
        "floor_number": 3,
        "total_floors": 10,
        "furnishing_status": "Semi-Furnished",
        "parking_available": 1,
        "lift_available": 1,
        "gated_community": 1,
    }
