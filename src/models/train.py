"""Model training and evaluation pipeline for Indian property price regression."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.config.settings import (
    ALL_FEATURE_COLUMNS,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data.load_data import load_processed_data
from src.data.prepare_features import build_canonical_dataset
from src.features.feature_pipeline import build_full_pipeline
from src.models.evaluate import compute_slice_metrics, evaluate_regression_model
from src.models.model_registry import save_model_artifact
from src.utils.logging_config import setup_logger

logger = setup_logger("train")


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split canonical dataset into train and test sets with stratified city representation."""
    logger.info("Splitting dataset into train and held-out test sets (80/20)...")
    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratify by city to preserve geographic distribution
    stratify_col = df["city"] if "city" in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )

    logger.info(f"Dataset split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows.")
    return X_train, X_test, y_train, y_test


def train_and_evaluate_all_models(
    processed_df: pd.DataFrame,
    save_best: bool = True,
) -> dict[str, Any]:
    """Train candidate models, compare metrics on held-out test data, and persist the champion model."""
    X_train, X_test, y_train, y_test = prepare_training_data(processed_df)

    candidates = {
        "Baseline_Median": DummyRegressor(strategy="median"),
        "Ridge_Regression": Ridge(alpha=10.0),
        "Hist_Gradient_Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=12,
            min_samples_leaf=15,
            random_state=RANDOM_STATE,
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=75,
            max_depth=14,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        logger.info(f"--- Training candidate model: {name} ---")
        use_log = name != "Baseline_Median"
        pipeline = build_full_pipeline(estimator, min_locality_freq=5, use_log_target=use_log)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_regression_model(y_test, y_pred, model_name=name)
        results[name] = metrics
        fitted_pipelines[name] = pipeline

    # Select best candidate (lowest MAE on test set among non-baseline models)
    competing_models = {k: v for k, v in results.items() if k != "Baseline_Median"}
    best_model_name = min(competing_models.keys(), key=lambda k: competing_models[k]["mae_inr"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    logger.info(f"Selected Champion Model: {best_model_name} (Test MAE: {best_metrics['mae_formatted']}, R2: {best_metrics['r2_score']:.4f})")

    # Perform slice error analysis on the champion model
    y_pred_best = best_pipeline.predict(X_test)
    eval_df = X_test.copy()
    eval_df["price_inr"] = y_test
    eval_df["pred_price_inr"] = y_pred_best
    slice_analysis = compute_slice_metrics(eval_df)

    # Save artifacts if requested
    if save_best:
        metadata = {
            "model_name": best_model_name,
            "version": "1.0.0",
            "model_type": str(candidates[best_model_name].__class__.__name__),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_metrics": best_metrics,
            "candidate_comparison": results,
            "slice_analysis": slice_analysis,
            "city_distribution": processed_df["city"].value_counts().to_dict(),
        }
        save_model_artifact(best_pipeline, metadata)

        # Generate markdown report
        generate_markdown_training_report(results, best_model_name, slice_analysis)

    return {
        "candidate_results": results,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "slice_analysis": slice_analysis,
        "best_pipeline": best_pipeline,
    }


def generate_markdown_training_report(
    candidate_results: dict[str, Any],
    best_model_name: str,
    slice_analysis: dict[str, Any],
) -> Path:
    """Generate and write training summary report to reports/."""
    report_file = REPORTS_DIR / "model_training_report.md"

    lines = [
        "# India House Price Prediction — Model Training & Evaluation Report",
        "",
        "## 1. Candidate Model Performance Comparison (Held-Out Test Set)",
        "",
        "| Model Name | Test MAE | Test RMSE | Median AE | Test R² | MAPE (%) |",
        "|---|---|---|---|---|---|",
    ]

    for name, m in candidate_results.items():
        mape_str = f"{m['mape_percentage']:.2f}%" if not np.isnan(m["mape_percentage"]) else "N/A"
        badge = " **(Champion)**" if name == best_model_name else ""
        lines.append(
            f"| `{name}`{badge} | {m['mae_formatted']} | {m['rmse_formatted']} | {m['median_ae_formatted']} | {m['r2_score']:.4f} | {mape_str} |"
        )

    lines.extend([
        "",
        "## 2. Slice Error Analysis (Champion Model)",
        "",
        "### A. Performance by City",
        "",
        "| City | Samples | MAE | Median AE | R² |",
        "|---|---|---|---|---|",
    ])

    for city, sm in slice_analysis.get("by_city", {}).items():
        lines.append(
            f"| **{city}** | {sm['sample_size']} | {sm['mae_formatted']} | {sm['median_ae_formatted']} | {sm['r2_score']:.4f} |"
        )

    lines.extend([
        "",
        "### B. Performance by Price Segment",
        "",
        "| Price Segment | Samples | MAE | Median AE | R² |",
        "|---|---|---|---|---|",
    ])

    for seg, sm in slice_analysis.get("by_price_segment", {}).items():
        lines.append(
            f"| **{seg}** | {sm['sample_size']} | {sm['mae_formatted']} | {sm['median_ae_formatted']} | {sm['r2_score']:.4f} |"
        )

    lines.extend([
        "",
        "### C. Performance by Property Type",
        "",
        "| Property Type | Samples | MAE | Median AE | R² |",
        "|---|---|---|---|---|",
    ])

    for ptype, sm in slice_analysis.get("by_property_type", {}).items():
        lines.append(
            f"| **{ptype}** | {sm['sample_size']} | {sm['mae_formatted']} | {sm['median_ae_formatted']} | {sm['r2_score']:.4f} |"
        )

    lines.append("")
    report_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved training report to {report_file}")
    return report_file


if __name__ == "__main__":
    try:
        df = load_processed_data()
    except FileNotFoundError:
        df = build_canonical_dataset()
    train_and_evaluate_all_models(df)
