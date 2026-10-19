"""Automated schema detection, column role inference, and dataset profiling."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_role: str
    non_null_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    sample_values: list[Any]


@dataclass
class DatasetSchemaReport:
    total_rows: int
    total_columns: int
    duplicate_rows: int
    column_profiles: dict[str, ColumnProfile]
    inferred_roles: dict[str, str | None]
    supports_valuation: bool
    missing_valuation_requirements: list[str]
    detected_city_column: str | None = None
    detected_area_column: str | None = None
    detected_price_column: str | None = None
    detected_bhk_column: str | None = None


ROLE_CANDIDATE_KEYWORDS = {
    "price": ["price", "cost", "amount", "valuation", "rate_inr", "price_inr", "target"],
    "city": ["city", "town", "metro", "region", "district"],
    "locality": ["locality", "location", "neighborhood", "area_name", "suburb", "address", "sector"],
    "bathrooms": ["bathrooms", "bathroom", "baths", "bath", "washrooms", "washroom", "toilets"],
    "bhk": ["bhk", "bedroom", "bedrooms", "bed_room", "bed_rooms", "beds", "size"],
    "area": ["area", "sqft", "sq_ft", "square_feet", "super_builtup", "carpet_area", "size_sqft", "total_sqft"],
    "floor": ["floor", "floor_number", "floornumber", "level"],
    "total_floors": ["total_floors", "totalfloors", "building_floors", "floors_count"],
    "furnishing": ["furnish", "furnishing", "furnishing_status", "furnished"],
    "parking": ["parking", "car_parking", "garage"],
    "property_type": ["property_type", "type", "property", "house_type"],
}


CANONICAL_ROLES = {
    "price_inr": "price",
    "area_sqft": "area",
    "bhk": "bhk",
    "bathrooms": "bathrooms",
    "city": "city",
    "locality": "locality",
    "property_type": "property_type",
    "furnishing_status": "furnishing",
    "parking_available": "parking",
    "floor_number": "floor",
    "total_floors": "total_floors",
}


def infer_column_role(col_name: str, sample_series: pd.Series) -> str:
    """Infer the semantic property role of a column based on name keywords and data content."""
    clean_name = col_name.lower().replace(" ", "_").replace("-", "_")

    # 1. Exact canonical column names have absolute highest priority
    if clean_name in CANONICAL_ROLES:
        return CANONICAL_ROLES[clean_name]

    # 2. Guard against false positive metadata/categorical columns
    if clean_name in ["price_type", "cost_type", "rate_type", "valuation_type"]:
        return "generic_attribute"
    if clean_name in ["area_type", "type", "house_type"]:
        return "property_type"
    if clean_name in ["source_id", "id", "listing_id"]:
        return "generic_attribute"

    # 3. Exact keyword match
    for role, keywords in ROLE_CANDIDATE_KEYWORDS.items():
        if clean_name in keywords:
            return role

    # 4. Substring match (skip for numeric roles if column is non-numeric or ending in _type/_id)
    is_numeric = pd.api.types.is_numeric_dtype(sample_series)
    for role, keywords in ROLE_CANDIDATE_KEYWORDS.items():
        if role in ["price", "area", "bhk", "bathrooms", "floor", "total_floors"]:
            if clean_name.endswith(("_type", "_id", "_status", "_name", "_unit")):
                continue
            if not is_numeric:
                # Check if it can be coerced to numeric with >50% non-null valid numbers
                coerced = pd.to_numeric(sample_series.dropna().head(20), errors="coerce")
                if coerced.isna().all():
                    continue

        if any(kw in clean_name for kw in keywords if len(kw) >= 3):
            return role

    # 5. Fallback heuristics based on dtype and cardinality
    if is_numeric and len(sample_series.dropna()) > 0:
        if sample_series.max() > 100_000:
            return "price"
        elif sample_series.between(1, 10).all():
            return "bhk"
        elif sample_series.max() > 100:
            return "area"

    return "generic_attribute"


def detect_dataset_schema(df: pd.DataFrame) -> DatasetSchemaReport:
    """Inspect and profile uploaded or loaded dataframe, returning structured schema findings."""
    total_rows, total_cols = df.shape
    dup_count = int(df.duplicated().sum())

    column_profiles = {}
    inferred_roles = {}
    role_priority = {}

    for col in df.columns:
        series = df[col]
        clean_name = col.lower().replace(" ", "_").replace("-", "_")
        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / total_rows * 100.0) if total_rows > 0 else 0.0
        role = infer_column_role(col, series)

        # Determine match strength priority
        priority = 0
        if clean_name in CANONICAL_ROLES:
            priority = 4
        elif any(clean_name == kw for kw in ROLE_CANDIDATE_KEYWORDS.get(role, [])):
            priority = 3
        elif any(kw in clean_name for kw in ROLE_CANDIDATE_KEYWORDS.get(role, [])):
            priority = 2
        else:
            priority = 1

        # Only overwrite an existing role assignment if the current candidate has strictly higher priority
        if role != "generic_attribute":
            existing_priority = role_priority.get(role, -1)
            if priority > existing_priority:
                inferred_roles[role] = col
                role_priority[role] = priority
        else:
            if "generic_attribute" not in inferred_roles:
                inferred_roles["generic_attribute"] = col

        samples = series.dropna().head(3).tolist()
        column_profiles[col] = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            inferred_role=role,
            non_null_count=int(series.notna().sum()),
            missing_count=missing_count,
            missing_percentage=round(missing_pct, 2),
            unique_count=int(series.nunique()),
            sample_values=samples,
        )

    # Check valuation model prerequisites
    missing_reqs = []
    if "city" not in inferred_roles:
        missing_reqs.append("City / Metropolitan Location")
    if "area" not in inferred_roles:
        missing_reqs.append("Property Area (Square Feet)")
    if "bhk" not in inferred_roles:
        missing_reqs.append("Bedroom Count (BHK)")

    supports_valuation = len(missing_reqs) == 0

    return DatasetSchemaReport(
        total_rows=total_rows,
        total_columns=total_cols,
        duplicate_rows=dup_count,
        column_profiles=column_profiles,
        inferred_roles=inferred_roles,
        supports_valuation=supports_valuation,
        missing_valuation_requirements=missing_reqs,
        detected_city_column=inferred_roles.get("city"),
        detected_area_column=inferred_roles.get("area"),
        detected_price_column=inferred_roles.get("price"),
        detected_bhk_column=inferred_roles.get("bhk"),
    )


def get_active_or_default_dataset() -> pd.DataFrame | None:
    """Safely retrieve active or cleaned DataFrame from session state, or load bundled dataset as default."""
    import streamlit as st

    if "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None:
        df = st.session_state["cleaned_df"]
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df

    if "active_df" in st.session_state and st.session_state["active_df"] is not None:
        df = st.session_state["active_df"]
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df

    try:
        from src.data.load_data import load_processed_data

        df = load_processed_data()
        st.session_state["active_df"] = df
        st.session_state["dataset_source"] = "bundled"
        return df
    except (FileNotFoundError, ValueError, OSError):
        return None

