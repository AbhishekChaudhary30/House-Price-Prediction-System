"""Data ingestion and loading functions for raw and processed datasets."""

from pathlib import Path

import pandas as pd

from src.config.settings import (
    PRIMARY_SUPPORTED_CITIES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data.data_sources import DATA_SOURCE_CATALOG, download_data_source
from src.utils.logging_config import setup_logger

logger = setup_logger("load_data")


def load_bengaluru_raw(force_download: bool = False) -> pd.DataFrame:
    """Load Bengaluru raw CSV dataset."""
    path = RAW_DATA_DIR / DATA_SOURCE_CATALOG["Bengaluru"].local_filename
    if not path.exists() or force_download:
        path = download_data_source("Bengaluru", force_download=force_download)
    df = pd.read_csv(path)
    logger.info(f"Loaded raw Bengaluru dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_delhi_raw(force_download: bool = False) -> pd.DataFrame:
    """Load Delhi raw CSV dataset."""
    path = RAW_DATA_DIR / DATA_SOURCE_CATALOG["Delhi"].local_filename
    if not path.exists() or force_download:
        path = download_data_source("Delhi", force_download=force_download)
    df = pd.read_csv(path)
    logger.info(f"Loaded raw Delhi dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_mumbai_raw(force_download: bool = False) -> pd.DataFrame:
    """Load Mumbai raw CSV dataset."""
    path = RAW_DATA_DIR / DATA_SOURCE_CATALOG["Mumbai"].local_filename
    if not path.exists() or force_download:
        path = download_data_source("Mumbai", force_download=force_download)
    df = pd.read_csv(path)
    logger.info(f"Loaded raw Mumbai dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_all_raw_datasets(force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Download and load all configured raw city datasets into memory."""
    datasets = {}
    for city in PRIMARY_SUPPORTED_CITIES:
        try:
            if city == "Bengaluru":
                datasets[city] = load_bengaluru_raw(force_download=force_download)
            elif city == "Delhi":
                datasets[city] = load_delhi_raw(force_download=force_download)
            elif city == "Mumbai":
                datasets[city] = load_mumbai_raw(force_download=force_download)
        except Exception as e:
            logger.error(f"Error loading dataset for city '{city}': {e}")
            raise
    return datasets


def load_processed_data(processed_path: Path | None = None) -> pd.DataFrame:
    """Load the harmonized, cleaned multi-city processed dataset."""
    target_path = processed_path or (PROCESSED_DATA_DIR / "india_housing_processed.csv")
    if not target_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found at: {target_path}. "
            "Please run the data preparation pipeline first (e.g., python -m src.data.prepare_features)."
        )
    df = pd.read_csv(target_path)
    logger.info(f"Loaded processed dataset from {target_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
