"""Model evaluation metrics, slice analysis, and performance reporting."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.utils.formatters import format_inr_short
from src.utils.logging_config import setup_logger

logger = setup_logger("evaluate")


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Calculate standard regression metrics in raw INR and Lakhs."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Ensure non-negative predictions for metrics calculation
    y_pred_arr = np.clip(y_pred_arr, a_min=0, a_max=None)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    med_ae = float(median_absolute_error(y_true_arr, y_pred_arr))

    # Calculate MAPE safely
    try:
        mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))
    except (ValueError, TypeError, ZeroDivisionError):
        mape = float(np.nan)

    metrics = {
        "model_name": model_name,
        "sample_size": len(y_true_arr),
        "mae_inr": mae,
        "mae_formatted": format_inr_short(mae),
        "rmse_inr": rmse,
        "rmse_formatted": format_inr_short(rmse),
        "median_ae_inr": med_ae,
        "median_ae_formatted": format_inr_short(med_ae),
        "r2_score": r2,
        "mape_percentage": mape * 100.0 if not np.isnan(mape) else np.nan,
    }

    logger.info(
        f"[{model_name}] MAE: {metrics['mae_formatted']} | "
        f"RMSE: {metrics['rmse_formatted']} | "
        f"MedAE: {metrics['median_ae_formatted']} | "
        f"R2: {r2:.4f} | "
        f"MAPE: {metrics['mape_percentage']:.2f}%"
    )

    return metrics


def compute_slice_metrics(
    df_eval: pd.DataFrame,
    y_true_col: str = "price_inr",
    y_pred_col: str = "pred_price_inr",
) -> dict[str, Any]:
    """Calculate error metrics segmented by city, property type, BHK, and price bracket."""
    eval_df = df_eval.copy()
    slices: dict[str, Any] = {"by_city": {}, "by_property_type": {}, "by_bhk": {}, "by_price_segment": {}}

    # Slice by City
    if "city" in eval_df.columns:
        for city, grp in eval_df.groupby("city"):
            if len(grp) >= 10:
                slices["by_city"][str(city)] = evaluate_regression_model(
                    grp[y_true_col], grp[y_pred_col], model_name=f"City_{city}"
                )

    # Slice by Property Type
    if "property_type" in eval_df.columns:
        for ptype, grp in eval_df.groupby("property_type"):
            if len(grp) >= 10:
                slices["by_property_type"][str(ptype)] = evaluate_regression_model(
                    grp[y_true_col], grp[y_pred_col], model_name=f"Type_{ptype}"
                )

    # Slice by BHK
    if "bhk" in eval_df.columns:
        for bhk_val, grp in eval_df.groupby("bhk"):
            if len(grp) >= 20 and bhk_val <= 5:
                slices["by_bhk"][f"{int(bhk_val)}_BHK"] = evaluate_regression_model(
                    grp[y_true_col], grp[y_pred_col], model_name=f"BHK_{bhk_val}"
                )

    # Slice by Price Segment
    # Budget: < 50 Lakhs
    # Mid-Market: 50L - 1.5 Cr
    # Premium: 1.5 Cr - 4 Cr
    # Luxury: > 4 Cr
    def get_price_segment(price: float) -> str:
        if price < 5_000_000:
            return "Budget (< 50L)"
        elif price < 15_000_000:
            return "Mid-Market (50L - 1.5Cr)"
        elif price < 40_000_000:
            return "Premium (1.5Cr - 4Cr)"
        else:
            return "Luxury (> 4Cr)"

    eval_df["price_segment"] = eval_df[y_true_col].apply(get_price_segment)
    for seg, grp in eval_df.groupby("price_segment"):
        if len(grp) >= 10:
            slices["by_price_segment"][str(seg)] = evaluate_regression_model(
                grp[y_true_col], grp[y_pred_col], model_name=f"Segment_{seg}"
            )

    return slices
