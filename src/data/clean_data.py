"""Data cleaning, normalization, and harmonization pipelines for Indian property datasets."""

import re

import pandas as pd

from src.config.settings import (
    MAX_AREA_SQFT,
    MAX_BATHROOMS,
    MAX_BHK,
    MAX_PRICE_INR,
    MIN_AREA_SQFT,
    MIN_BATHROOMS,
    MIN_BHK,
    MIN_PRICE_INR,
)
from src.utils.formatters import normalize_location_name, parse_area_to_sqft
from src.utils.logging_config import setup_logger

logger = setup_logger("clean_data")


def extract_bhk(size_str: str | None) -> int | None:
    """Extract BHK integer count from size string (e.g. '2 BHK', '3 Bedroom', '1 RK')."""
    if pd.isna(size_str) or not str(size_str).strip():
        return None

    s = str(size_str).strip().lower()
    if "rk" in s:
        return 1

    match = re.search(r"(\d+)", s)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def clean_bengaluru_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw Bengaluru dataset into canonical format."""
    logger.info(f"Cleaning Bengaluru dataset: initial shape {df.shape}")
    cleaned = df.copy()

    # City identifier
    cleaned["city"] = "Bengaluru"

    # Locality normalization
    cleaned["locality"] = cleaned["location"].apply(normalize_location_name)

    # Area parsing
    cleaned["area_sqft"] = cleaned["total_sqft"].apply(parse_area_to_sqft)

    # BHK extraction
    cleaned["bhk"] = cleaned["size"].apply(extract_bhk)

    # Bathrooms
    cleaned["bathrooms"] = pd.to_numeric(cleaned["bath"], errors="coerce")

    # Property type mapping
    def map_prop_type(area_type: str) -> str:
        s = str(area_type).lower()
        if "plot" in s:
            return "Independent House"
        return "Apartment"

    cleaned["property_type"] = cleaned["area_type"].apply(map_prop_type)

    # Price conversion: raw Bengaluru prices are listed in Lakhs INR (e.g., 39.07 -> ₹39,07,000)
    cleaned["price_inr"] = pd.to_numeric(cleaned["price"], errors="coerce") * 100_000.0

    # Synthetic / typical features for missing structural details
    cleaned["furnishing_status"] = "Semi-Furnished"
    cleaned["floor_number"] = 2
    cleaned["total_floors"] = 4
    cleaned["parking_available"] = cleaned["bathrooms"].apply(
        lambda b: 1 if b >= 2 else 0
    )
    cleaned["lift_available"] = cleaned["property_type"].apply(
        lambda t: 1 if t == "Apartment" else 0
    )
    cleaned["gated_community"] = cleaned["society"].notna().astype(int)

    cleaned["price_type"] = "asking_price"
    cleaned["source_id"] = "bengaluru_listings_kaggle"

    return cleaned


def clean_delhi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw Delhi dataset into canonical format."""
    logger.info(f"Cleaning Delhi dataset: initial shape {df.shape}")
    cleaned = df.copy()

    # Drop potential target leakage column if present in source
    if "Per_Sqft" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Per_Sqft"])

    cleaned["city"] = "Delhi"
    cleaned["locality"] = cleaned["Locality"].apply(normalize_location_name)
    cleaned["area_sqft"] = pd.to_numeric(cleaned["Area"], errors="coerce")
    cleaned["bhk"] = pd.to_numeric(cleaned["BHK"], errors="coerce")
    cleaned["bathrooms"] = pd.to_numeric(cleaned["Bathroom"], errors="coerce")

    # Furnishing standardization
    def map_furnishing(f: str) -> str:
        s = str(f).lower().strip()
        if "semi" in s:
            return "Semi-Furnished"
        elif "unfurnished" in s:
            return "Unfurnished"
        elif "furnished" in s:
            return "Furnished"
        return "Semi-Furnished"

    cleaned["furnishing_status"] = cleaned["Furnishing"].apply(map_furnishing)

    # Property type mapping
    def map_delhi_type(t: str) -> str:
        s = str(t).lower().strip()
        if "builder" in s:
            return "Builder Floor"
        return "Apartment"

    cleaned["property_type"] = cleaned["Type"].apply(map_delhi_type)

    # Raw Delhi price is already in numeric INR
    cleaned["price_inr"] = pd.to_numeric(cleaned["Price"], errors="coerce")

    # Parking
    cleaned["parking_available"] = cleaned["Parking"].apply(
        lambda p: 1 if pd.notna(p) and float(p) >= 1 else 0
    )
    cleaned["floor_number"] = 2
    cleaned["total_floors"] = 4
    cleaned["lift_available"] = cleaned["property_type"].apply(
        lambda t: 1 if t == "Apartment" else 0
    )
    cleaned["gated_community"] = 1

    cleaned["price_type"] = "asking_price"
    cleaned["source_id"] = "delhi_magicbricks_listings"

    return cleaned


def clean_mumbai_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw Mumbai dataset into canonical format."""
    logger.info(f"Cleaning Mumbai dataset: initial shape {df.shape}")
    cleaned = df.copy()

    cleaned["city"] = "Mumbai"
    cleaned["locality"] = cleaned["Location"].apply(normalize_location_name)

    # Carpet Area / Area / SuperBuiltUp Area selection
    # Prefer Carpet Area if positive, then Area, then SuperBuiltUp Area
    def resolve_mumbai_area(row) -> float | None:
        for col in ["Carpet Area", "Area", "SuperBuiltUp Area"]:
            val = row.get(col)
            if pd.notna(val) and float(val) > 50:
                return float(val)
        return None

    cleaned["area_sqft"] = cleaned.apply(resolve_mumbai_area, axis=1)

    cleaned["bhk"] = pd.to_numeric(cleaned["Bedrooms"], errors="coerce")
    cleaned["bathrooms"] = pd.to_numeric(cleaned["Bathrooms"], errors="coerce")

    # Floor number handling
    cleaned["floor_number"] = (
        pd.to_numeric(cleaned["FloorNumber"], errors="coerce")
        .fillna(2)
        .clip(lower=0, upper=60)
        .astype(int)
    )
    cleaned["total_floors"] = cleaned["floor_number"].apply(lambda f: max(f + 2, 7))

    # Furnishing
    def map_mumbai_furnishing(f: str) -> str:
        s = str(f).lower().strip()
        if "semi" in s:
            return "Semi-Furnished"
        elif "unfurnished" in s:
            return "Unfurnished"
        elif "furnished" in s:
            return "Furnished"
        return "Semi-Furnished"

    cleaned["furnishing_status"] = cleaned["Furnishing"].apply(map_mumbai_furnishing)

    # Property type
    def map_mumbai_prop(p: str) -> str:
        s = str(p).lower().strip()
        if "villa" in s:
            return "Villa"
        elif "house" in s:
            return "Independent House"
        return "Apartment"

    cleaned["property_type"] = cleaned["Property"].apply(map_mumbai_prop)

    # Price conversion: raw Mumbai dataset prices are in Lakhs INR (e.g., 158.5 -> ₹1,58,50,000)
    cleaned["price_inr"] = pd.to_numeric(cleaned["Price"], errors="coerce") * 100_000.0

    # Amenities
    cleaned["parking_available"] = cleaned["Parking"].apply(
        lambda p: 1 if pd.notna(p) and float(p) >= 1 else 0
    )
    cleaned["lift_available"] = cleaned["floor_number"].apply(
        lambda f: 1 if f >= 2 else 0
    )
    cleaned["gated_community"] = cleaned["parking_available"].apply(
        lambda p: 1 if p == 1 else 0
    )

    cleaned["price_type"] = "asking_price"
    cleaned["source_id"] = "mumbai_real_estate_appraisal"

    return cleaned


def filter_plausible_records(df: pd.DataFrame) -> pd.DataFrame:
    """Filter records using domain-defensible boundaries and statistical consistency checks."""
    initial_count = len(df)
    logger.info(f"Filtering dataset: initial count = {initial_count}")

    # Drop missing target or critical features
    valid = df.dropna(subset=["price_inr", "area_sqft", "bhk", "city"]).copy()

    # Price boundaries (₹3 Lakhs to ₹50 Crores)
    valid = valid[
        (valid["price_inr"] >= MIN_PRICE_INR) & (valid["price_inr"] <= MAX_PRICE_INR)
    ]

    # Area boundaries (150 sqft to 25,000 sqft)
    valid = valid[
        (valid["area_sqft"] >= MIN_AREA_SQFT) & (valid["area_sqft"] <= MAX_AREA_SQFT)
    ]

    # BHK boundaries (1 to 10)
    valid = valid[(valid["bhk"] >= MIN_BHK) & (valid["bhk"] <= MAX_BHK)]

    # Bathrooms boundaries (1 to 10)
    if "bathrooms" in valid.columns:
        bath_series = pd.to_numeric(valid["bathrooms"], errors="coerce")
        bhk_fallback = pd.to_numeric(valid["bhk"], errors="coerce")
        valid["bathrooms"] = (
            bath_series.fillna(bhk_fallback)
            .astype(float)
            .clip(lower=float(MIN_BATHROOMS), upper=float(MAX_BATHROOMS))
        )

    # Implausible price per sqft check:
    # Extremely cheap (< ₹800/sqft) or absurdly expensive (> ₹300,000/sqft) are data errors in Indian real estate
    pps = valid["price_inr"] / valid["area_sqft"]
    valid = valid[(pps >= 800.0) & (pps <= 300_000.0)]

    # Implausible area per BHK check:
    # A 1-BHK with 10,000 sqft or a 5-BHK with 150 sqft is a data error
    area_per_bhk = valid["area_sqft"] / valid["bhk"]
    valid = valid[(area_per_bhk >= 100.0) & (area_per_bhk <= 5000.0)]

    # Floor number consistency: floor_number <= total_floors
    if "floor_number" in valid.columns and "total_floors" in valid.columns:
        mask_floor = valid["floor_number"] > valid["total_floors"]
        valid.loc[mask_floor, "total_floors"] = (
            valid.loc[mask_floor, "floor_number"] + 1
        )

    # Deduplicate exact property duplicates
    subset_cols = ["city", "locality", "area_sqft", "bhk", "bathrooms", "price_inr"]
    existing_subset = [c for c in subset_cols if c in valid.columns]
    valid = valid.drop_duplicates(subset=existing_subset)

    removed = initial_count - len(valid)
    logger.info(
        f"Filtering complete: retained {len(valid)} rows ({removed} invalid/outlier rows removed, {removed / initial_count * 100:.1f}%)"
    )
    return valid


def clean_housing_data(raw_data_dict: dict) -> pd.DataFrame:
    """Standardize and harmonize all raw city datasets into one unified canonical dataframe."""
    frames = []

    if "Bengaluru" in raw_data_dict:
        df_b = clean_bengaluru_data(raw_data_dict["Bengaluru"])
        frames.append(df_b)

    if "Delhi" in raw_data_dict:
        df_d = clean_delhi_data(raw_data_dict["Delhi"])
        frames.append(df_d)

    if "Mumbai" in raw_data_dict:
        df_m = clean_mumbai_data(raw_data_dict["Mumbai"])
        frames.append(df_m)

    if not frames:
        raise ValueError("No valid city dataframes found to clean and combine.")

    combined = pd.concat(frames, ignore_index=True)
    filtered = filter_plausible_records(combined)

    return filtered
