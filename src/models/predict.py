"""Inference functions for generating raw model predictions."""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import ALL_FEATURE_COLUMNS
from src.utils.logging_config import setup_logger

logger = setup_logger("predict")


def predict_property_price(
    pipeline: Pipeline,
    input_data: pd.DataFrame | dict,
) -> np.ndarray:
    """Generate price prediction in Indian Rupees (INR) using fitted pipeline."""
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Ensure all required feature columns are present
    missing = [c for c in ALL_FEATURE_COLUMNS if c not in df_input.columns]
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    # Generate raw prediction
    raw_pred = pipeline.predict(df_input[ALL_FEATURE_COLUMNS])

    # Ensure non-negative output
    preds_inr = np.clip(raw_pred, a_min=0.0, a_max=None)
    return preds_inr
