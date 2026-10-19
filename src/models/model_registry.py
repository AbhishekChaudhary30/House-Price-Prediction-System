"""Model artifact persistence, versioning, serialization, and deserialization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.config.settings import (
    ALL_FEATURE_COLUMNS,
    ARTIFACTS_DIR,
    METADATA_ARTIFACT_NAME,
    MODEL_ARTIFACT_NAME,
    TARGET_COLUMN,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("model_registry")


def save_model_artifact(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    model_name: str | None = None,
    metadata_name: str | None = None,
) -> tuple[Path, Path]:
    """Serialize fitted model pipeline and corresponding metadata to disk."""
    model_file = ARTIFACTS_DIR / (model_name or MODEL_ARTIFACT_NAME)
    meta_file = ARTIFACTS_DIR / (metadata_name or METADATA_ARTIFACT_NAME)

    # Ensure timestamp and versioning fields exist
    metadata["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["target_column"] = TARGET_COLUMN
    metadata["feature_columns"] = ALL_FEATURE_COLUMNS

    logger.info(f"Saving model pipeline to {model_file}...")
    joblib.dump(pipeline, model_file, compress=3)

    logger.info(f"Saving model metadata to {meta_file}...")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model pipeline and metadata successfully saved.")
    return model_file, meta_file


def load_model_artifact(
    model_name: str | None = None,
    metadata_name: str | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load serialized model pipeline and metadata safely."""
    model_file = ARTIFACTS_DIR / (model_name or MODEL_ARTIFACT_NAME)
    meta_file = ARTIFACTS_DIR / (metadata_name or METADATA_ARTIFACT_NAME)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {model_file}. "
            "Please train the model first by running: python -m src.models.train"
        )

    logger.info(f"Loading model pipeline from {model_file}...")
    pipeline = joblib.load(model_file)

    metadata = {}
    if meta_file.exists():
        logger.info(f"Loading metadata from {meta_file}...")
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.warning(f"Metadata file {meta_file} not found; proceeding with empty metadata.")

    return pipeline, metadata
