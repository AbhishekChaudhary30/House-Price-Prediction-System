"""Orchestrator to load raw data, clean, validate, construct location registry, and save processed artifacts."""

import json
from typing import Any

import pandas as pd

from src.config.settings import (
    ALL_FEATURE_COLUMNS,
    ARTIFACTS_DIR,
    LOCATION_REGISTRY_NAME,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)
from src.data.clean_data import clean_housing_data
from src.data.load_data import load_all_raw_datasets
from src.data.validate_data import validate_dataset_schema
from src.utils.logging_config import setup_logger

logger = setup_logger("prepare_features")


def build_location_registry(df: pd.DataFrame, min_locality_observations: int = 3) -> dict[str, Any]:
    """Build a data-driven coverage registry of supported cities, localities, and property characteristics."""
    registry: dict[str, Any] = {
        "cities": {},
        "total_records": len(df),
        "min_locality_observations": min_locality_observations,
    }

    for city, city_group in df.groupby("city"):
        city_name = str(city)
        locality_stats = {}

        for locality, loc_group in city_group.groupby("locality"):
            loc_name = str(locality)
            obs_count = len(loc_group)
            is_supported = obs_count >= min_locality_observations

            locality_stats[loc_name] = {
                "observation_count": obs_count,
                "is_supported": is_supported,
                "median_price_inr": float(loc_group["price_inr"].median()),
                "min_price_inr": float(loc_group["price_inr"].min()),
                "max_price_inr": float(loc_group["price_inr"].max()),
                "median_area_sqft": float(loc_group["area_sqft"].median()),
                "supported_bhk": sorted([int(x) for x in loc_group["bhk"].unique()]),
                "property_types": sorted(loc_group["property_type"].unique().tolist()),
            }

        registry["cities"][city_name] = {
            "observation_count": len(city_group),
            "median_price_inr": float(city_group["price_inr"].median()),
            "median_area_sqft": float(city_group["area_sqft"].median()),
            "total_localities": len(locality_stats),
            "supported_localities_count": sum(1 for v in locality_stats.values() if v["is_supported"]),
            "localities": locality_stats,
            "property_types": sorted(city_group["property_type"].unique().tolist()),
            "supported_bhk": sorted([int(x) for x in city_group["bhk"].unique()]),
        }

    return registry


def build_canonical_dataset(force_download: bool = False) -> pd.DataFrame:
    """Execute end-to-end data pipeline to produce processed dataset and location registry."""
    logger.info("Beginning data acquisition and preparation pipeline...")

    # Step 1: Load raw datasets
    raw_datasets = load_all_raw_datasets(force_download=force_download)

    # Step 2: Clean and harmonize
    cleaned_df = clean_housing_data(raw_datasets)

    # Step 3: Validate schema and domain constraints
    report = validate_dataset_schema(cleaned_df, require_target=True)
    if not report.is_valid:
        raise ValueError(f"Cleaned dataset failed validation: {report.errors}")

    logger.info(f"Validation successful: {report.row_count} rows across {len(report.city_counts)} cities.")

    # Step 4: Build location registry
    location_registry = build_location_registry(cleaned_df)
    registry_path = ARTIFACTS_DIR / LOCATION_REGISTRY_NAME
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(location_registry, f, indent=2)
    logger.info(f"Saved location registry to {registry_path}")

    # Step 5: Save processed dataset with canonical columns only
    canonical_cols = ALL_FEATURE_COLUMNS + [TARGET_COLUMN, "price_type", "source_id"]
    save_df = cleaned_df[[c for c in canonical_cols if c in cleaned_df.columns]].copy()
    processed_path = PROCESSED_DATA_DIR / "india_housing_processed.csv"
    save_df.to_csv(processed_path, index=False)
    logger.info(f"Saved processed dataset ({save_df.shape[0]} rows, {save_df.shape[1]} cols) to {processed_path}")

    return save_df


if __name__ == "__main__":
    build_canonical_dataset(force_download=False)
