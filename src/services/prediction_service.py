"""Prediction service layer: validation, location normalization, inference, and structured responses."""

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import (
    ARTIFACTS_DIR,
    CITY_NORMALIZATION_MAP,
    LOCATION_REGISTRY_NAME,
    MAX_AREA_SQFT,
    MAX_BATHROOMS,
    MAX_BHK,
    MAX_FLOOR,
    MAX_TOTAL_FLOORS,
    MIN_AREA_SQFT,
    MIN_BATHROOMS,
    MIN_BHK,
    MIN_FLOOR,
    MIN_TOTAL_FLOORS,
    PRIMARY_SUPPORTED_CITIES,
    VALID_FURNISHING_STATUSES,
)
from src.data.data_sources import DATA_SOURCE_CATALOG
from src.models.model_registry import load_model_artifact
from src.models.predict import predict_property_price
from src.utils.formatters import (
    format_inr,
    format_inr_short,
    normalize_location_name,
    parse_area_to_sqft,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("prediction_service")


@dataclass
class PredictionRequest:
    city: str
    locality: str
    area_value: float
    area_unit: str = "sqft"
    bhk: int = 2
    bathrooms: int = 2
    property_type: str = "Apartment"
    furnishing_status: str = "Semi-Furnished"
    floor_number: int = 2
    total_floors: int = 4
    parking_available: bool = True
    lift_available: bool = True
    gated_community: bool = True


@dataclass
class PredictionResponse:
    success: bool
    estimated_price_inr: float = 0.0
    formatted_price: str = "₹0"
    formatted_short: str = "₹0"
    price_per_sqft_inr: float = 0.0
    formatted_price_per_sqft: str = "₹0"
    lower_bound_inr: float = 0.0
    upper_bound_inr: float = 0.0
    formatted_interval: str = "₹0 - ₹0"
    city: str = ""
    locality: str = ""
    normalized_area_sqft: float = 0.0
    location_support_status: str = "UNSUPPORTED"
    confidence_tier: str = "NONE"
    locality_observations: int = 0
    model_version: str = "1.0.0"
    data_source_reference: str = ""
    limitations: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


class HousingPredictionService:
    """Service to validate inputs, apply location coverage checks, and serve model predictions."""

    def __init__(self, pipeline: Pipeline | None = None, metadata: dict[str, Any] | None = None):
        if pipeline is not None:
            self.pipeline = pipeline
            self.metadata = metadata or {}
        else:
            self.pipeline, self.metadata = load_model_artifact()

        self.location_registry = self._load_location_registry()

    def _load_location_registry(self) -> dict[str, Any]:
        reg_file = ARTIFACTS_DIR / LOCATION_REGISTRY_NAME
        if reg_file.exists():
            with open(reg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"Location registry not found at {reg_file}; initializing empty.")
        return {"cities": {}}

    def normalize_city(self, city_input: str) -> str:
        """Resolve city aliases (e.g. Bangalore -> Bengaluru, Bombay -> Mumbai)."""
        key = str(city_input).strip().lower()
        return CITY_NORMALIZATION_MAP.get(key, normalize_location_name(city_input))

    def validate_request(self, req: PredictionRequest) -> list[str]:
        """Validate input parameters against physical domain constraints."""
        errors: list[str] = []

        # Area validation
        area_sqft = parse_area_to_sqft(req.area_value, default_unit=req.area_unit)
        if area_sqft is None or area_sqft < MIN_AREA_SQFT or area_sqft > MAX_AREA_SQFT:
            errors.append(
                f"Property area ({req.area_value} {req.area_unit}) converts to "
                f"{area_sqft or 0:.0f} sqft, which falls outside plausible residential range [{MIN_AREA_SQFT:.0f}, {MAX_AREA_SQFT:.0f} sqft]."
            )

        # BHK validation
        if req.bhk < MIN_BHK or req.bhk > MAX_BHK:
            errors.append(f"BHK must be between {MIN_BHK} and {MAX_BHK}, got {req.bhk}.")

        # Bathrooms validation
        if req.bathrooms < MIN_BATHROOMS or req.bathrooms > MAX_BATHROOMS:
            errors.append(f"Bathrooms count must be between {MIN_BATHROOMS} and {MAX_BATHROOMS}, got {req.bathrooms}.")

        # Floor validation
        if req.floor_number < MIN_FLOOR or req.floor_number > MAX_FLOOR:
            errors.append(f"Floor number must be between {MIN_FLOOR} and {MAX_FLOOR}, got {req.floor_number}.")

        if req.total_floors < MIN_TOTAL_FLOORS or req.total_floors > MAX_TOTAL_FLOORS:
            errors.append(f"Total building floors must be between {MIN_TOTAL_FLOORS} and {MAX_TOTAL_FLOORS}, got {req.total_floors}.")

        if req.floor_number > req.total_floors:
            errors.append(
                f"Floor number ({req.floor_number}) cannot exceed total building floors ({req.total_floors})."
            )

        # Categorical sanity checks
        if req.furnishing_status not in VALID_FURNISHING_STATUSES:
            errors.append(
                f"Invalid furnishing status '{req.furnishing_status}'. Accepted: {VALID_FURNISHING_STATUSES}"
            )

        return errors

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        """Process validated property input, evaluate location support, and return price estimate."""
        # 1. Input Validation
        val_errors = self.validate_request(req)
        if val_errors:
            return PredictionResponse(
                success=False,
                validation_errors=val_errors,
                limitations=["Prediction aborted due to input validation failures."],
            )

        # 2. Location Normalization & Distribution Check
        city_norm = self.normalize_city(req.city)
        locality_norm = normalize_location_name(req.locality)
        area_sqft = parse_area_to_sqft(req.area_value, default_unit=req.area_unit)

        cities_data = self.location_registry.get("cities", {})

        # Check City Support
        if city_norm not in PRIMARY_SUPPORTED_CITIES and city_norm not in cities_data:
            return PredictionResponse(
                success=False,
                city=city_norm,
                locality=locality_norm,
                location_support_status="UNSUPPORTED_CITY",
                confidence_tier="UNSUPPORTED",
                limitations=[
                    f"City '{city_norm}' is not represented in the verified training dataset.",
                    f"Currently supported cities: {', '.join(PRIMARY_SUPPORTED_CITIES)}.",
                    "The system refuses to substitute an arbitrary city to avoid fabricating prices.",
                ],
                validation_errors=[f"Unsupported city '{city_norm}'."],
            )

        # Check Locality Support
        city_info = cities_data.get(city_norm, {})
        locs_info = city_info.get("localities", {})
        loc_meta = locs_info.get(locality_norm)

        locality_obs = 0
        if loc_meta:
            locality_obs = loc_meta.get("observation_count", 0)
            if locality_obs >= 8:
                loc_status = "SUPPORTED_LOCALITY"
                confidence = "HIGH"
            elif locality_obs >= 3:
                loc_status = "SUPPORTED_LOCALITY"
                confidence = "MODERATE"
            else:
                loc_status = "LOW_SAMPLE_LOCALITY"
                confidence = "LOW"
        else:
            loc_status = "CITY_LEVEL_FALLBACK"
            confidence = "MODERATE"

        # 3. Assemble Inference Record
        row_dict = {
            "city": city_norm,
            "locality": locality_norm,
            "property_type": req.property_type,
            "bhk": req.bhk,
            "bathrooms": req.bathrooms,
            "area_sqft": area_sqft,
            "floor_number": req.floor_number,
            "total_floors": req.total_floors,
            "furnishing_status": req.furnishing_status,
            "parking_available": 1 if req.parking_available else 0,
            "lift_available": 1 if req.lift_available else 0,
            "gated_community": 1 if req.gated_community else 0,
        }
        df_input = pd.DataFrame([row_dict])

        # 4. Generate Machine Learning Prediction
        try:
            preds = predict_property_price(self.pipeline, df_input)
            est_price = float(preds[0])
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"Inference error during model prediction: {e}")
            return PredictionResponse(
                success=False,
                validation_errors=[f"Inference error: {e!s}"],
                limitations=["An internal error occurred while evaluating the feature pipeline."],
            )

        # 5. Calculate Defensible Uncertainty Interval
        # Based on test-set median absolute percentage error (~18% for high confidence, ~25% for city fallback)
        error_margin = 0.18 if confidence == "HIGH" else 0.25
        lower_bound = max(est_price * (1.0 - error_margin), 300_000.0)
        upper_bound = est_price * (1.0 + error_margin)

        price_per_sqft = est_price / area_sqft if area_sqft > 0 else 0.0

        # Citations and limitations
        src_meta = DATA_SOURCE_CATALOG.get(city_norm)
        data_citation = src_meta.citation if src_meta else "Curated Indian residential listing dataset."

        limitations = [
            "Estimate represents asking price valuation based on historical property listings.",
            "Actual transaction prices may vary due to legal clearance, negotiation, builder reputation, and specific micro-amenities.",
        ]
        if loc_status == "CITY_LEVEL_FALLBACK":
            limitations.append(
                f"Locality '{locality_norm}' was not directly observed in the training split. "
                f"Estimate utilizes broader '{city_norm}' city-level pricing baselines."
            )

        return PredictionResponse(
            success=True,
            estimated_price_inr=est_price,
            formatted_price=format_inr(est_price),
            formatted_short=format_inr_short(est_price),
            price_per_sqft_inr=price_per_sqft,
            formatted_price_per_sqft=f"{format_inr(price_per_sqft)} / sq.ft.",
            lower_bound_inr=lower_bound,
            upper_bound_inr=upper_bound,
            formatted_interval=f"{format_inr_short(lower_bound)} - {format_inr_short(upper_bound)}",
            city=city_norm,
            locality=locality_norm,
            normalized_area_sqft=area_sqft,
            location_support_status=loc_status,
            confidence_tier=confidence,
            locality_observations=locality_obs,
            model_version=self.metadata.get("version", "1.0.0"),
            data_source_reference=data_citation,
            limitations=limitations,
        )


_service_instance: HousingPredictionService | None = None


def get_prediction_service() -> HousingPredictionService:
    """Singleton getter for prediction service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = HousingPredictionService()
    return _service_instance
