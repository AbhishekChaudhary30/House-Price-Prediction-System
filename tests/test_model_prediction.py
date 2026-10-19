"""Tests for model training, inference, persistence, and sensitivity."""

import numpy as np
from sklearn.linear_model import Ridge

from src.config.settings import ALL_FEATURE_COLUMNS, TARGET_COLUMN
from src.features.feature_pipeline import build_full_pipeline
from src.models.evaluate import evaluate_regression_model
from src.models.model_registry import load_model_artifact, save_model_artifact
from src.models.predict import predict_property_price


def test_pipeline_trains_and_evaluates_on_fixture(sample_valid_dataframe):
    X = sample_valid_dataframe[ALL_FEATURE_COLUMNS]
    y = sample_valid_dataframe[TARGET_COLUMN]

    pipeline = build_full_pipeline(Ridge(alpha=1.0), min_locality_freq=1, use_log_target=True)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert (preds > 0).all()

    metrics = evaluate_regression_model(y, preds, model_name="TestRidge")
    assert "mae_inr" in metrics
    assert "r2_score" in metrics
    assert metrics["mae_inr"] >= 0


def test_model_responds_to_area_increase(sample_valid_dataframe):
    """Domain consistency check: increasing property area should increase estimated price."""
    X = sample_valid_dataframe[ALL_FEATURE_COLUMNS]
    y = sample_valid_dataframe[TARGET_COLUMN]

    pipeline = build_full_pipeline(Ridge(alpha=1.0), min_locality_freq=1, use_log_target=True)
    pipeline.fit(X, y)

    base_input = X.iloc[[0]].copy()
    larger_input = X.iloc[[0]].copy()
    larger_input["area_sqft"] = base_input["area_sqft"].values[0] * 2.0

    pred_base = predict_property_price(pipeline, base_input)[0]
    pred_larger = predict_property_price(pipeline, larger_input)[0]

    assert pred_larger > pred_base


def test_model_persistence_and_reload(sample_valid_dataframe, tmp_path, monkeypatch):
    from src.models import model_registry

    monkeypatch.setattr(model_registry, "ARTIFACTS_DIR", tmp_path)

    X = sample_valid_dataframe[ALL_FEATURE_COLUMNS]
    y = sample_valid_dataframe[TARGET_COLUMN]
    pipeline = build_full_pipeline(Ridge(alpha=1.0), min_locality_freq=1, use_log_target=True)
    pipeline.fit(X, y)

    meta = {"version": "test_1.0", "name": "TestModel"}
    model_file, meta_file = save_model_artifact(
        pipeline, meta, model_name="test_model.joblib", metadata_name="test_meta.json"
    )
    assert model_file.exists()
    assert meta_file.exists()

    loaded_pipe, loaded_meta = load_model_artifact(
        model_name="test_model.joblib", metadata_name="test_meta.json"
    )

    pred_original = predict_property_price(pipeline, X.iloc[[0]])
    pred_loaded = predict_property_price(loaded_pipe, X.iloc[[0]])

    assert np.isclose(pred_original, pred_loaded).all()
    assert loaded_meta["name"] == "TestModel"
