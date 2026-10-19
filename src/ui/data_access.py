"""Shared data-access layer for Visual Explorer and Query & Filter Studio."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import streamlit as st

from src.data.load_data import load_processed_data

REQUIRED_COLUMNS = [
    "city",
    "locality",
    "property_type",
    "bhk",
    "bathrooms",
    "area_sqft",
    "price_inr",
    "furnishing_status",
]

NUMERIC_COLUMNS = [
    "bhk",
    "bathrooms",
    "area_sqft",
    "price_inr",
    "floor_number",
    "total_floors",
]


@dataclass
class DatasetDiagnosticInfo:
    total_records: int = 0
    usable_records: int = 0
    missing_columns: list[str] = field(default_factory=list)
    source: str = "unknown"
    is_valid: bool = False
    error_message: str | None = None
    column_stats: dict[str, Any] = field(default_factory=dict)


def get_validated_explorer_dataset() -> tuple[
    pd.DataFrame | None, DatasetDiagnosticInfo
]:
    """Retrieve and validate the shared housing dataset for visualization and querying.

    Ensures:
    - Consistent source loading across visualization and filter studios.
    - Explicit column validation and non-destructive type enforcement.
    - Diagnostic reporting of total vs. valid records.
    - Graceful error handling without silent failures.
    """
    source_name = "unknown"
    raw_df = None

    if "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None:
        cand = st.session_state["cleaned_df"]
        if isinstance(cand, pd.DataFrame) and len(cand) > 0:
            raw_df = cand
            source_name = "Automated Cleaned Dataset"

    if (
        raw_df is None
        and "active_df" in st.session_state
        and st.session_state["active_df"] is not None
    ):
        cand = st.session_state["active_df"]
        if isinstance(cand, pd.DataFrame) and len(cand) > 0:
            raw_df = cand
            source_name = f"Active Workspace ({st.session_state.get('dataset_source', 'uploaded')})"

    if raw_df is None:
        try:
            raw_df = load_processed_data()
            source_name = "Bundled Multi-City Dataset (55,543 records)"
            st.session_state["active_df"] = raw_df
            st.session_state["dataset_source"] = "bundled"
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError) as e:
            return None, DatasetDiagnosticInfo(
                total_records=0,
                usable_records=0,
                missing_columns=REQUIRED_COLUMNS,
                source="unloaded",
                is_valid=False,
                error_message=f"Failed to load dataset: {e}",
            )

    if raw_df is None or len(raw_df) == 0:
        return None, DatasetDiagnosticInfo(
            total_records=0,
            usable_records=0,
            missing_columns=REQUIRED_COLUMNS,
            source="empty",
            is_valid=False,
            error_message="Active dataset contains 0 records.",
        )

    # Work on a defensive copy to prevent unintended mutations
    df = raw_df.copy()

    # Column name normalization (strip whitespace, lowercase lookup map)
    existing_cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Schema mapping check
    col_mapping = {}
    missing = []
    for req in REQUIRED_COLUMNS:
        if req in df.columns:
            continue
        req_clean = req.lower()
        if req_clean in existing_cols_lower:
            col_mapping[existing_cols_lower[req_clean]] = req
        elif req == "price_inr":
            # Check price aliases
            for alias in ["price", "cost", "amount", "rate_inr"]:
                if alias in existing_cols_lower:
                    col_mapping[existing_cols_lower[alias]] = "price_inr"
                    break
            else:
                missing.append(req)
        elif req == "area_sqft":
            for alias in ["area", "sqft", "total_sqft", "size_sqft"]:
                if alias in existing_cols_lower:
                    col_mapping[existing_cols_lower[alias]] = "area_sqft"
                    break
            else:
                missing.append(req)
        elif req == "bhk":
            for alias in ["bedroom", "bedrooms", "bed_room", "beds"]:
                if alias in existing_cols_lower:
                    col_mapping[existing_cols_lower[alias]] = "bhk"
                    break
            else:
                missing.append(req)
        else:
            missing.append(req)

    if col_mapping:
        df = df.rename(columns=col_mapping)

    # Ensure numeric types are actually numeric
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure categorical columns are clean strings
    for col in ["city", "locality", "property_type", "furnishing_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Calculate usable records (records with valid price, area, and city)
    crit_cols = [c for c in ["price_inr", "area_sqft", "city"] if c in df.columns]
    if crit_cols:
        usable_count = int(df.dropna(subset=crit_cols).shape[0])
    else:
        usable_count = len(df)

    is_valid = len(missing) == 0 and usable_count > 0
    error_msg = None
    if not is_valid:
        if missing:
            error_msg = f"Missing required columns: {', '.join(missing)}"
        elif usable_count == 0:
            error_msg = "No records contain valid price, area, and city values."

    diag = DatasetDiagnosticInfo(
        total_records=len(df),
        usable_records=usable_count,
        missing_columns=missing,
        source=source_name,
        is_valid=is_valid,
        error_message=error_msg,
        column_stats={
            "columns_present": list(df.columns),
            "cities": df["city"].unique().tolist() if "city" in df.columns else [],
            "bhk_range": (int(df["bhk"].min()), int(df["bhk"].max()))
            if "bhk" in df.columns and df["bhk"].notna().any()
            else None,
        },
    )

    return df, diag
