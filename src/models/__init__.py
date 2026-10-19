"""Model training, evaluation, inference, and serialization registry."""
from src.models.evaluate import compute_slice_metrics, evaluate_regression_model
from src.models.model_registry import load_model_artifact, save_model_artifact
from src.models.predict import predict_property_price
from src.models.train import train_and_evaluate_all_models

__all__ = [
    "compute_slice_metrics",
    "evaluate_regression_model",
    "load_model_artifact",
    "predict_property_price",
    "save_model_artifact",
    "train_and_evaluate_all_models",
]
