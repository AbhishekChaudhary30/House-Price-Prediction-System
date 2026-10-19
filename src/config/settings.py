import os
from pathlib import Path

# Suppress loky/wmic physical core warning on modern Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, DOCS_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Target column name
TARGET_COLUMN = "price_inr"

# Canonical feature schema
NUMERICAL_FEATURES: list[str] = [
    "bhk",
    "bathrooms",
    "area_sqft",
    "floor_number",
    "total_floors",
]

CATEGORICAL_FEATURES: list[str] = [
    "city",
    "locality",
    "property_type",
    "furnishing_status",
]

BINARY_FEATURES: list[str] = [
    "parking_available",
    "lift_available",
    "gated_community",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Engineered features created in feature pipeline
ENGINEERED_NUMERICAL_FEATURES = [
    "area_per_bhk",
    "floor_ratio",
    "bath_to_bhk_ratio",
]

ALL_MODEL_NUMERICAL = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

# Allowed / canonical categories
VALID_FURNISHING_STATUSES = [
    "Unfurnished",
    "Semi-Furnished",
    "Furnished",
]

VALID_PROPERTY_TYPES = [
    "Apartment",
    "Builder Floor",
    "Independent House",
    "Villa",
    "Penthouse",
]

# Validation limits for Indian residential properties
MIN_AREA_SQFT = 150.0
MAX_AREA_SQFT = 25000.0

MIN_BHK = 1
MAX_BHK = 10

MIN_BATHROOMS = 1
MAX_BATHROOMS = 10

MIN_FLOOR = 0
MAX_FLOOR = 100

MIN_TOTAL_FLOORS = 1
MAX_TOTAL_FLOORS = 120

# Price boundaries (in INR)
# Min: 3 Lakhs (₹3,00,000)
# Max: 50 Crores (₹50,00,00,000)
MIN_PRICE_INR = 300_000.0
MAX_PRICE_INR = 500_000_000.0

# City Normalization Mapping
CITY_NORMALIZATION_MAP: dict[str, str] = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "blr": "Bengaluru",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "navi mumbai": "Mumbai",
    "thane": "Mumbai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "delhi ncr": "Delhi",
    "ncr": "Delhi",
    "noida": "Delhi",
    "gurgaon": "Delhi",
    "gurugram": "Delhi",
    "ghaziabad": "Delhi",
    "faridabad": "Delhi",
    "pune": "Pune",
    "hyderabad": "Hyderabad",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
}

# Supported cities with primary training coverage in v1.0
PRIMARY_SUPPORTED_CITIES = ["Bengaluru", "Mumbai", "Delhi"]

# Data Source URLs (verified open benchmark & research repositories)
DATA_SOURCE_URLS = {
    "Bengaluru": "https://raw.githubusercontent.com/tejas05in/pwskills-Data_Science_Masters/master/machine%20learning/assignments/Bengaluru_House_Data.csv",
    "Delhi": "https://raw.githubusercontent.com/rohit077/Delhi-Flat-Price-Prediction/master/MagicBricks.csv",
    "Mumbai": "https://raw.githubusercontent.com/ak1997/Comparison-of-Ensemble-Models-Using-Real-Estate-Appraisal/master/Mumbai.csv",
}

# Artifact file names
MODEL_ARTIFACT_NAME = "house_price_pipeline_v1.joblib"
METADATA_ARTIFACT_NAME = "model_metadata_v1.json"
LOCATION_REGISTRY_NAME = "location_registry_v1.json"
