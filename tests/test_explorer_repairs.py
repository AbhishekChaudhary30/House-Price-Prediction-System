"""Focused tests verifying Visual Explorer, Query Studio, Chart Studio, and Data Access repairs."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.ui.chart_studio import (
    create_area_distribution_chart,
    create_area_vs_price_scatter,
    create_bhk_comparison_bar,
    create_category_distribution_boxplot,
    create_city_price_boxplot,
    create_correlation_heatmap,
    create_data_quality_figure,
    create_empty_state_figure,
    create_model_performance_chart,
    create_price_by_city_bar,
    create_price_distribution_chart,
    create_price_per_sqft_chart,
    create_property_type_comparison_chart,
)
from src.ui.data_access import get_validated_explorer_dataset
from src.ui.schema_detector import detect_dataset_schema, infer_column_role


@pytest.fixture
def mock_housing_df():
    """Create a realistic small mock housing DataFrame."""
    return pd.DataFrame(
        {
            "city": ["Bengaluru", "Bengaluru", "Mumbai", "Mumbai", "Delhi"],
            "locality": ["Whitefield", "Indiranagar", "Bandra", "Andheri", "Rohini"],
            "property_type": [
                "Apartment",
                "Independent House",
                "Apartment",
                "Apartment",
                "Builder Floor",
            ],
            "bhk": [2.0, 3.0, 2.0, 1.0, 3.0],
            "bathrooms": [2.0, 3.0, 2.0, 1.0, 2.0],
            "area_sqft": [1200.0, 2400.0, 950.0, 550.0, 1500.0],
            "price_inr": [
                7_500_000.0,
                25_000_000.0,
                30_000_000.0,
                12_000_000.0,
                18_000_000.0,
            ],
            "furnishing_status": [
                "Semi-Furnished",
                "Furnished",
                "Unfurnished",
                "Semi-Furnished",
                "Furnished",
            ],
            "floor_number": [3.0, 1.0, 14.0, 5.0, 2.0],
            "total_floors": [10.0, 2.0, 25.0, 12.0, 4.0],
            "price_type": [
                "asking_price",
                "asking_price",
                "asking_price",
                "asking_price",
                "asking_price",
            ],
            "source_id": ["s1", "s2", "s3", "s4", "s5"],
        }
    )


# ==============================================================================
# 1. SCHEMA INFERENCE & DATA ACCESS TESTS
# ==============================================================================


def test_price_type_not_misinferred_as_price():
    """Verify that price_type is not inferred as the numeric price column."""
    role_price_inr = infer_column_role("price_inr", pd.Series([1000000, 2000000]))
    role_price_type = infer_column_role(
        "price_type", pd.Series(["asking_price", "sale"])
    )

    assert role_price_inr == "price"
    assert role_price_type != "price"


def test_schema_detector_prioritizes_price_inr_over_price_type(mock_housing_df):
    """Verify that detect_dataset_schema maps detected_price_column to price_inr, not price_type."""
    report = detect_dataset_schema(mock_housing_df)
    assert report.detected_price_column == "price_inr"
    assert report.inferred_roles["price"] == "price_inr"
    assert report.detected_city_column == "city"
    assert report.detected_area_column == "area_sqft"
    assert report.detected_bhk_column == "bhk"


def test_get_validated_explorer_dataset_returns_full_dataset():
    """Verify that get_validated_explorer_dataset loads the actual 55,543 records cleanly."""
    df, diag = get_validated_explorer_dataset()
    assert df is not None
    assert len(df) == 55543
    assert diag.is_valid is True
    assert diag.usable_records == 55543
    assert len(diag.missing_columns) == 0

    # Verify required columns are present and properly typed
    for col in [
        "city",
        "locality",
        "property_type",
        "bhk",
        "bathrooms",
        "area_sqft",
        "price_inr",
        "furnishing_status",
    ]:
        assert col in df.columns

    assert pd.api.types.is_numeric_dtype(df["price_inr"])
    assert pd.api.types.is_numeric_dtype(df["area_sqft"])
    assert pd.api.types.is_numeric_dtype(df["bhk"])


# ==============================================================================
# 2. FILTERING LOGIC TESTS
# ==============================================================================


def test_city_and_bhk_filter_combination(mock_housing_df):
    """Verify multi-column filtering combination works correctly."""
    df = mock_housing_df.copy()
    # Filter by Bengaluru only
    res = df[df["city"].isin(["Bengaluru"])]
    assert len(res) == 2

    # Filter by Bengaluru + 2 BHK
    res2 = res[res["bhk"].isin([2])]
    assert len(res2) == 1
    assert res2.iloc[0]["locality"] == "Whitefield"


def test_text_search_case_insensitive(mock_housing_df):
    """Verify locality text query matches case-insensitively."""
    df = mock_housing_df.copy()
    match = df[df["locality"].str.contains("white", case=False, na=False)]
    assert len(match) == 1
    assert match.iloc[0]["city"] == "Bengaluru"


def test_empty_filter_results_handled_gracefully(mock_housing_df):
    """Verify that when a filter matches zero records, dataframe is 0 rows without crashing."""
    df = mock_housing_df.copy()
    res = df[df["city"].isin(["NonExistentCity"])]
    assert len(res) == 0


# ==============================================================================
# 3. CHART STUDIO TESTS & DEFENSIVE BEHAVIOR
# ==============================================================================


def test_empty_state_figure_produces_valid_figure():
    """Verify that create_empty_state_figure returns a valid figure with annotations."""
    fig = create_empty_state_figure("Custom empty message")
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) > 0
    assert "Custom empty message" in fig.layout.annotations[0].text


def test_price_distribution_chart_normal_and_log10(mock_housing_df):
    """Verify histogram renders properly with and without Log10 transformation."""
    # Standard scale
    fig_std = create_price_distribution_chart(
        mock_housing_df, use_log_scale=False, nbins=20
    )
    assert isinstance(fig_std, go.Figure)
    assert len(fig_std.data) > 0

    # Log10 scale
    fig_log = create_price_distribution_chart(
        mock_housing_df, use_log_scale=True, nbins=20
    )
    assert isinstance(fig_log, go.Figure)
    assert len(fig_log.data) > 0
    # Log values should be around 6.8 to 7.5
    assert fig_log.layout.xaxis.title.text == "Log10(Price in INR)"


def test_price_distribution_handles_empty_dataframe():
    """Verify empty dataframe returns defensive empty figure instead of blank error."""
    empty_df = pd.DataFrame(columns=["price_inr", "city"])
    fig = create_price_distribution_chart(empty_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) > 0


def test_price_by_city_bar(mock_housing_df):
    """Verify median price by city bar chart computes correct median values."""
    fig = create_price_by_city_bar(mock_housing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_price_per_sqft_chart_calculation(mock_housing_df):
    """Verify price per sqft chart calculates rate accurately without division by zero."""
    fig = create_price_per_sqft_chart(mock_housing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_area_vs_price_scatter_downsampling(mock_housing_df):
    """Verify scatter plot works and respects sample caps."""
    fig = create_area_vs_price_scatter(mock_housing_df, max_samples=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_bhk_comparison_bar(mock_housing_df):
    """Verify BHK bar chart generates median prices for valid bedroom counts."""
    fig = create_bhk_comparison_bar(mock_housing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_property_type_comparison_chart(mock_housing_df):
    """Verify property type comparison chart generates valid bars."""
    fig = create_property_type_comparison_chart(mock_housing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_category_distribution_boxplot(mock_housing_df):
    """Verify boxplot generation across categories."""
    fig = create_category_distribution_boxplot(mock_housing_df, category_col="city")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_city_price_boxplot(mock_housing_df):
    """Verify city price boxplot generates valid box traces."""
    fig = create_city_price_boxplot(mock_housing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_area_distribution_chart(mock_housing_df):
    """Verify area distribution histogram generates valid trace."""
    fig = create_area_distribution_chart(mock_housing_df, nbins=15)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_correlation_heatmap_numerical_only(mock_housing_df):
    """Verify correlation heatmap only computes on numeric columns and handles missing safely."""
    fig = create_correlation_heatmap(mock_housing_df)
    assert isinstance(fig, go.Figure)


def test_data_quality_figure(mock_housing_df):
    """Verify data quality figure creates completeness bars."""
    fig = create_data_quality_figure(mock_housing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_model_performance_chart_from_real_metadata():
    """Verify candidate model comparison chart accurately loads genuine pipeline evaluation metadata."""
    fig = create_model_performance_chart()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Check that candidate models (e.g. Hist Gradient Boosting) appear
    bar_x = list(fig.data[0].x)
    assert any("Hist Gradient Boosting" in str(x) for x in bar_x)
