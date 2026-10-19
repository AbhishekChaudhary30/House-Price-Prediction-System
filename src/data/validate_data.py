"""Data validation functions and data quality reporting."""

from dataclasses import dataclass, field

import pandas as pd

from src.config.settings import (
    ALL_FEATURE_COLUMNS,
    MAX_PRICE_INR,
    MIN_PRICE_INR,
    TARGET_COLUMN,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("validate_data")


@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    column_count: int
    missing_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    target_summary: dict[str, float] = field(default_factory=dict)
    city_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Data Validation Report ({status}) ===",
            f"Total Rows: {self.row_count}, Columns: {self.column_count}",
        ]
        if self.city_counts:
            lines.append("City Distribution: " + ", ".join(f"{k}: {v}" for k, v in self.city_counts.items()))
        if self.errors:
            lines.append("Errors:\n  - " + "\n  - ".join(self.errors))
        if self.warnings:
            lines.append("Warnings:\n  - " + "\n  - ".join(self.warnings))
        return "\n".join(lines)


def validate_dataset_schema(
    df: pd.DataFrame,
    required_features: list[str] | None = None,
    require_target: bool = True,
) -> ValidationReport:
    """Validate DataFrame against canonical housing schema, types, and boundaries."""
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return ValidationReport(
            is_valid=False,
            row_count=0,
            column_count=0,
            errors=["Input is not a valid pandas DataFrame."],
        )

    row_count, col_count = df.shape
    if row_count == 0:
        return ValidationReport(
            is_valid=False,
            row_count=0,
            column_count=col_count,
            errors=["DataFrame is empty."],
        )

    # Check target
    target_summary = {}
    if require_target:
        if TARGET_COLUMN not in df.columns:
            errors.append(f"Missing required target column: '{TARGET_COLUMN}'")
        else:
            target_series = df[TARGET_COLUMN]
            if not pd.api.types.is_numeric_dtype(target_series):
                errors.append(f"Target column '{TARGET_COLUMN}' must be numeric, got {target_series.dtype}.")
            else:
                nan_target = target_series.isna().sum()
                if nan_target > 0:
                    errors.append(f"Target column '{TARGET_COLUMN}' contains {nan_target} NaN values.")

                neg_target = (target_series <= 0).sum()
                if neg_target > 0:
                    errors.append(f"Target column '{TARGET_COLUMN}' contains {neg_target} non-positive values.")

                out_bounds = ((target_series < MIN_PRICE_INR) | (target_series > MAX_PRICE_INR)).sum()
                if out_bounds > 0:
                    warnings.append(
                        f"{out_bounds} records have prices outside recommended range [{MIN_PRICE_INR:.0f}, {MAX_PRICE_INR:.0f}]."
                    )

                clean_target = target_series.dropna()
                if len(clean_target) > 0:
                    target_summary = {
                        "min": float(clean_target.min()),
                        "median": float(clean_target.median()),
                        "mean": float(clean_target.mean()),
                        "max": float(clean_target.max()),
                        "std": float(clean_target.std()),
                    }

    # Check required features
    check_features = required_features or ALL_FEATURE_COLUMNS
    missing_cols = [c for c in check_features if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {missing_cols}")

    # Check numeric types
    for num_col in ["bhk", "bathrooms", "area_sqft"]:
        if num_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[num_col]):
                errors.append(f"Feature '{num_col}' must be numeric, got {df[num_col].dtype}")
            else:
                neg_vals = (df[num_col] <= 0).sum()
                if neg_vals > 0:
                    errors.append(f"Feature '{num_col}' contains {neg_vals} non-positive values.")

    # Check duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"Dataset contains {dup_count} duplicate rows.")

    # Missing counts
    missing_counts = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    for c, cnt in missing_counts.items():
        ratio = cnt / row_count
        if ratio > 0.4:
            warnings.append(f"Feature '{c}' has high missingness: {ratio * 100:.1f}% ({cnt} rows).")

    # City counts
    city_counts = {}
    if "city" in df.columns:
        city_counts = df["city"].value_counts().to_dict()

    is_valid = len(errors) == 0
    report = ValidationReport(
        is_valid=is_valid,
        row_count=row_count,
        column_count=col_count,
        missing_counts=missing_counts,
        errors=errors,
        warnings=warnings,
        target_summary=target_summary,
        city_counts=city_counts,
    )

    if not is_valid:
        logger.error(f"Dataset validation failed with {len(errors)} errors: {errors}")
    else:
        logger.info(f"Dataset validation passed ({row_count} rows, {col_count} columns).")

    return report
