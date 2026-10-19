"""Tests for Part 2 capabilities: schema detection, automated cleaning, analytics, and chart studio."""

import pandas as pd
import plotly.graph_objects as go

from src.ui.analytics_engine import run_one_click_analysis
from src.ui.auto_cleaner import run_automated_cleaning
from src.ui.chart_studio import (
    create_area_vs_price_scatter,
    create_bhk_comparison_bar,
    create_city_price_boxplot,
    create_correlation_heatmap,
    create_price_distribution_chart,
)
from src.ui.schema_detector import detect_dataset_schema


def test_schema_detector_identifies_roles_correctly(sample_valid_dataframe):
    report = detect_dataset_schema(sample_valid_dataframe)
    assert report.total_rows == len(sample_valid_dataframe)
    assert report.supports_valuation is True
    assert report.detected_city_column == "city"
    assert report.detected_area_column == "area_sqft"
    assert report.detected_price_column == "price_inr"
    assert report.detected_bhk_column == "bhk"


def test_schema_detector_flags_missing_requirements():
    incomplete_df = pd.DataFrame(
        {
            "generic_col1": [1, 2, 3],
            "generic_col2": ["A", "B", "C"],
        }
    )
    report = detect_dataset_schema(incomplete_df)
    assert report.supports_valuation is False
    assert len(report.missing_valuation_requirements) > 0


def test_automated_cleaning_removes_duplicates_and_empty_rows(sample_valid_dataframe):
    corrupted_df = sample_valid_dataframe.copy()
    # Add duplicate row
    corrupted_df = pd.concat(
        [corrupted_df, sample_valid_dataframe.iloc[[0]]], ignore_index=True
    )
    # Add empty row
    corrupted_df.loc[len(corrupted_df)] = [float("nan")] * len(corrupted_df.columns)

    roles = {
        "locality": "locality",
        "city": "city",
        "area": "area_sqft",
        "price": "price_inr",
    }
    cleaned_df, report = run_automated_cleaning(corrupted_df, roles)

    assert report.original_row_count == len(sample_valid_dataframe) + 2
    assert report.duplicates_removed >= 1
    assert len(cleaned_df) == len(sample_valid_dataframe)


def test_automatic_analysis_generates_narratives_and_summaries(sample_valid_dataframe):
    insights = run_one_click_analysis(sample_valid_dataframe)
    assert insights.total_records == len(sample_valid_dataframe)
    assert "median_formatted" in insights.target_summary
    assert len(insights.narrative_insights) > 0
    assert len(insights.city_breakdown) > 0
    assert len(insights.bhk_breakdown) > 0


def test_plotly_chart_builders_produce_valid_figures(sample_valid_dataframe):
    fig_dist = create_price_distribution_chart(
        sample_valid_dataframe, use_log_scale=False
    )
    assert isinstance(fig_dist, go.Figure)

    fig_box = create_city_price_boxplot(sample_valid_dataframe)
    assert isinstance(fig_box, go.Figure)

    fig_scatter = create_area_vs_price_scatter(sample_valid_dataframe)
    assert isinstance(fig_scatter, go.Figure)

    fig_bhk = create_bhk_comparison_bar(sample_valid_dataframe)
    assert isinstance(fig_bhk, go.Figure)

    fig_corr = create_correlation_heatmap(sample_valid_dataframe)
    assert isinstance(fig_corr, go.Figure)


def test_get_active_or_default_dataset_handles_dataframe_safely():
    import streamlit as st

    from src.ui.schema_detector import get_active_or_default_dataset

    # Clean state
    st.session_state.pop("cleaned_df", None)
    st.session_state.pop("active_df", None)

    # When both are None, loads default processed dataset
    df = get_active_or_default_dataset()
    assert df is not None
    assert len(df) > 0

    # When active_df is set, retrieves active_df without ambiguous boolean errors
    custom_df = pd.DataFrame(
        {"city": ["Bengaluru"], "area_sqft": [1200], "price_inr": [7500000]}
    )
    st.session_state["active_df"] = custom_df
    retrieved = get_active_or_default_dataset()
    assert len(retrieved) == 1

    # When cleaned_df is present, prefers cleaned_df
    cleaned_custom = pd.DataFrame(
        {"city": ["Mumbai"], "area_sqft": [900], "price_inr": [15000000]}
    )
    st.session_state["cleaned_df"] = cleaned_custom
    retrieved_cleaned = get_active_or_default_dataset()
    assert retrieved_cleaned.iloc[0]["city"] == "Mumbai"
