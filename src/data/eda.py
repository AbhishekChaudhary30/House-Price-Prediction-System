"""Exploratory Data Analysis (EDA) module to generate and save statistical charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.settings import REPORTS_DIR
from src.utils.formatters import format_inr_short
from src.utils.logging_config import setup_logger

logger = setup_logger("eda")


def run_exploratory_data_analysis(df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> dict:
    """Generate meaningful EDA plots and summary statistics from the processed dataset."""
    logger.info("Executing Exploratory Data Analysis (EDA)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_summary = {}

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"

    # 1. Price Distribution (Raw INR & Log Scale)
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    price_cr = df["price_inr"] / 10_000_000  # Convert to Crores for clean visualization
    sns.histplot(price_cr, bins=40, kde=True, ax=ax1, color="#2563EB")
    ax1.set_title("Property Price Distribution (in Crores INR)")
    ax1.set_xlabel("Price (₹ Crores)")
    ax1.set_ylabel("Listing Count")

    sns.histplot(np.log1p(df["price_inr"]), bins=40, kde=True, ax=ax2, color="#059669")
    ax2.set_title("Log-Transformed Target Price: log(1 + Price)")
    ax2.set_xlabel("Log(Price INR)")
    ax2.set_ylabel("Listing Count")
    plt.tight_layout()
    chart1_path = output_dir / "eda_price_distribution.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {chart1_path}")

    # 2. Price by City Boxplot
    plt.figure(figsize=(9, 6))
    sns.boxplot(
        data=df,
        x="city",
        y=price_cr,
        palette=["#3B82F6", "#10B981", "#F59E0B"],
        showfliers=False,
    )
    plt.title("Price Distribution by City (Outliers hidden for clarity)")
    plt.xlabel("City")
    plt.ylabel("Price (₹ Crores)")
    plt.tight_layout()
    chart2_path = output_dir / "eda_price_by_city.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {chart2_path}")

    # 3. Area vs Price Scatter Plot by City
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(min(len(df), 3000), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="area_sqft",
        y=sample_df["price_inr"] / 10_000_000,
        hue="city",
        alpha=0.6,
        palette=["#2563EB", "#059669", "#D97706"],
    )
    plt.title("Property Area (sq.ft) vs Price (₹ Crores)")
    plt.xlabel("Property Area (sq.ft)")
    plt.ylabel("Price (₹ Crores)")
    plt.tight_layout()
    chart3_path = output_dir / "eda_area_vs_price.png"
    plt.savefig(chart3_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {chart3_path}")

    # 4. Correlation Heatmap
    num_cols = ["area_sqft", "bhk", "bathrooms", "floor_number", "total_floors", "price_inr"]
    corr_df = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    chart4_path = output_dir / "eda_correlation_matrix.png"
    plt.savefig(chart4_path, dpi=200)
    plt.close()
    logger.info(f"Saved: {chart4_path}")

    stats_summary["total_records"] = len(df)
    stats_summary["median_price_inr"] = float(df["price_inr"].median())
    stats_summary["median_price_formatted"] = format_inr_short(df["price_inr"].median())
    stats_summary["mean_area_sqft"] = float(df["area_sqft"].mean())
    stats_summary["cities"] = df["city"].value_counts().to_dict()

    return stats_summary


if __name__ == "__main__":
    from src.data.load_data import load_processed_data
    df_proc = load_processed_data()
    run_exploratory_data_analysis(df_proc)
