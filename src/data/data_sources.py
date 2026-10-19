"""Data source catalog, registry, and ingestion abstractions for Indian property data."""

from dataclasses import dataclass
from pathlib import Path

import requests

from src.config.settings import DATA_SOURCE_URLS, RAW_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger("data_sources")


@dataclass(frozen=True)
class DataSourceMetadata:
    source_id: str
    city: str
    price_type: str  # 'asking_price', 'transaction_price', or 'synthetic'
    is_synthetic: bool
    license: str
    citation: str
    collection_period: str
    url: str
    local_filename: str
    known_limitations: str


DATA_SOURCE_CATALOG: dict[str, DataSourceMetadata] = {
    "Bengaluru": DataSourceMetadata(
        source_id="bengaluru_listings_kaggle",
        city="Bengaluru",
        price_type="asking_price",
        is_synthetic=False,
        license="CC BY-SA 4.0 / Open Access",
        citation="Bengaluru House Price Data, Kaggle (Amitabha Joy) / 99acres real-estate listing aggregation.",
        collection_period="2016-2018 Listings",
        url=DATA_SOURCE_URLS["Bengaluru"],
        local_filename="bengaluru_raw.csv",
        known_limitations=(
            "Asking prices advertised on portals, not registered stamp duty deeds. "
            "May carry a 5-15% negotiation discount compared to final deed price. "
            "Floor number and total building floors are not explicitly recorded in this raw feed."
        ),
    ),
    "Delhi": DataSourceMetadata(
        source_id="delhi_magicbricks_listings",
        city="Delhi",
        price_type="asking_price",
        is_synthetic=False,
        license="MIT / Open Data for Educational Use",
        citation="Delhi House Price Dataset from MagicBricks portal listings (Rohit Gupta).",
        collection_period="2019-2020 Listings",
        url=DATA_SOURCE_URLS["Delhi"],
        local_filename="delhi_raw.csv",
        known_limitations=(
            "Captures residential flats, builder floors, and apartments in Delhi NCR. "
            "Contains advertised listing prices. Sample size (1,259 records) is smaller than Mumbai/Bengaluru."
        ),
    ),
    "Mumbai": DataSourceMetadata(
        source_id="mumbai_real_estate_appraisal",
        city="Mumbai",
        price_type="asking_price",
        is_synthetic=False,
        license="Open Access / Academic Portfolio Research",
        citation="Mumbai Real Estate Appraisal Dataset (Ashish Kumar / Mumbai Housing Portal Listings).",
        collection_period="2018-2021 Listings",
        url=DATA_SOURCE_URLS["Mumbai"],
        local_filename="mumbai_raw.csv",
        known_limitations=(
            "Covers Mumbai Metropolitan Region (MMR) including Mumbai City, Mumbai Suburban, Thane, and Navi Mumbai. "
            "Prices are listed asking prices. Contains carpet area, built-up area, and floor numbers."
        ),
    ),
}


def get_datasource_metadata(city: str) -> DataSourceMetadata | None:
    """Retrieve metadata for a specific city's data source."""
    return DATA_SOURCE_CATALOG.get(city)


def download_data_source(city: str, force_download: bool = False) -> Path:
    """Download raw dataset for a city if not already present."""
    if city not in DATA_SOURCE_CATALOG:
        raise ValueError(f"Unknown city source: '{city}'. Supported: {list(DATA_SOURCE_CATALOG.keys())}")

    meta = DATA_SOURCE_CATALOG[city]
    dest_path = RAW_DATA_DIR / meta.local_filename

    if dest_path.exists() and not force_download and dest_path.stat().st_size > 1000:
        logger.info(f"Using cached raw dataset for {city} at {dest_path}")
        return dest_path

    logger.info(f"Downloading raw dataset for {city} from {meta.url}...")
    try:
        response = requests.get(meta.url, timeout=30)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        logger.info(f"Saved {city} dataset ({dest_path.stat().st_size / 1024:.1f} KB) to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download dataset for {city}: {e}")
        raise
