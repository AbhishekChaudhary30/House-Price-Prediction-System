"""Automated data cleaning pipeline, quality audit, and before-and-after reporting."""

from dataclasses import dataclass, field

import pandas as pd

from src.data.clean_data import filter_plausible_records
from src.utils.formatters import normalize_location_name, parse_area_to_sqft


@dataclass
class CleaningAuditReport:
    original_row_count: int
    final_row_count: int
    original_col_count: int
    final_col_count: int
    duplicates_removed: int
    missing_values_handled: int
    outliers_removed: int
    modifications: list[str] = field(default_factory=list)
    remaining_warnings: list[str] = field(default_factory=list)


def run_automated_cleaning(
    raw_df: pd.DataFrame,
    detected_roles: dict[str, str],
) -> tuple[pd.DataFrame, CleaningAuditReport]:
    """Execute non-destructive automated cleaning pipeline and produce transparent audit report."""
    orig_rows, orig_cols = raw_df.shape
    cleaned = raw_df.copy()
    modifications = []
    warnings = []

    # 1. Drop completely empty rows
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(how="all")
    empty_dropped = before_drop - len(cleaned)
    if empty_dropped > 0:
        modifications.append(f"Removed {empty_dropped} completely blank rows.")

    # 2. Standardize column names (strip whitespace)
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # 3. Deduplicate exact duplicate records
    dup_count = int(cleaned.duplicated().sum())
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates()
        modifications.append(
            f"Detected and eliminated {dup_count} exact duplicate rows."
        )

    # 4. Normalize text and categorical columns
    locality_col = detected_roles.get("locality")
    if locality_col and locality_col in cleaned.columns:
        cleaned[locality_col] = cleaned[locality_col].apply(normalize_location_name)
        modifications.append(
            f"Standardized text casing and removed extraneous characters in '{locality_col}'."
        )

    city_col = detected_roles.get("city")
    if city_col and city_col in cleaned.columns:
        cleaned[city_col] = cleaned[city_col].apply(normalize_location_name)
        modifications.append(f"Normalized city names in '{city_col}'.")

    # 5. Clean area values
    area_col = detected_roles.get("area")
    if area_col and area_col in cleaned.columns:
        cleaned["area_sqft_cleaned"] = cleaned[area_col].apply(parse_area_to_sqft)
        modifications.append(
            f"Parsed area measurements and ranges in '{area_col}' into numeric sq.ft."
        )

    # 6. Clean numeric columns & domain bounds
    price_col = detected_roles.get("price")
    if (
        not price_col or price_col not in cleaned.columns
    ) and "price_inr" in cleaned.columns:
        price_col = "price_inr"

    if (
        not area_col or "area_sqft_cleaned" not in cleaned.columns
    ) and "area_sqft" in cleaned.columns:
        cleaned["area_sqft_cleaned"] = pd.to_numeric(
            cleaned["area_sqft"], errors="coerce"
        )

    bhk_col = detected_roles.get("bhk") or ("bhk" if "bhk" in cleaned.columns else None)
    city_col = detected_roles.get("city") or (
        "city" if "city" in cleaned.columns else None
    )

    # If canonical columns match our schema, apply defensible boundary filter
    outliers_count = 0
    if price_col and "area_sqft_cleaned" in cleaned.columns:
        temp_eval = cleaned.copy()
        temp_eval["area_sqft"] = temp_eval["area_sqft_cleaned"]
        temp_eval["price_inr"] = pd.to_numeric(temp_eval[price_col], errors="coerce")
        if bhk_col and bhk_col in temp_eval.columns:
            temp_eval["bhk"] = pd.to_numeric(temp_eval[bhk_col], errors="coerce")
        if city_col and city_col in temp_eval.columns:
            temp_eval["city"] = temp_eval[city_col]

        before_filter = len(temp_eval)
        try:
            valid_subset = filter_plausible_records(temp_eval)
            if len(valid_subset) > 0:
                cleaned = cleaned.loc[valid_subset.index].copy()
                outliers_count = before_filter - len(cleaned)
                if outliers_count > 0:
                    modifications.append(
                        f"Removed {outliers_count} implausible records (outlier prices, extreme area-to-BHK ratios)."
                    )
            else:
                warnings.append(
                    "Outlier boundary checks eliminated all records; retained original records to prevent empty dataset."
                )
        except (ValueError, KeyError, TypeError) as exc:
            warnings.append(
                f"Outlier filtering skipped due to schema discrepancy: {exc}"
            )

    # Cleaned area column promotion
    if "area_sqft_cleaned" in cleaned.columns:
        cleaned["area_sqft"] = cleaned["area_sqft_cleaned"]
        cleaned = cleaned.drop(columns=["area_sqft_cleaned"])

    final_rows, final_cols = cleaned.shape
    total_handled = orig_rows - final_rows

    if final_rows == 0:
        warnings.append(
            "Dataset has 0 rows remaining after cleaning. Please verify input data quality."
        )

    report = CleaningAuditReport(
        original_row_count=orig_rows,
        final_row_count=final_rows,
        original_col_count=orig_cols,
        final_col_count=final_cols,
        duplicates_removed=dup_count,
        missing_values_handled=total_handled,
        outliers_removed=outliers_count,
        modifications=modifications,
        remaining_warnings=warnings,
    )

    return cleaned, report
