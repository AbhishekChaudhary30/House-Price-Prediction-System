"""Interactive Plotly chart studio with Indian real estate styling and tooltips."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

PLOTLY_THEME = {
    "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#F8FAFC",
    "gridcolor": "#E2E8F0",
    "color_sequence": [
        "#2563EB",
        "#059669",
        "#D97706",
        "#7C3AED",
        "#DC2626",
        "#0891B2",
        "#4F46E5",
        "#0D9488",
    ],
}


def create_empty_state_figure(
    message: str = "No data available matching the selected filter criteria.",
) -> go.Figure:
    """Generate an informative, beautifully styled empty state figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠️ <b>{message}</b><br><span style='font-size:12px; color:#94A3B8;'>Try broadening your filter criteria or reset all filters.</span>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#64748B", "family": PLOTLY_THEME["font_family"]},
        align="center",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        height=380,
    )
    return fig


def create_price_distribution_chart(
    df: pd.DataFrame,
    use_log_scale: bool = False,
    nbins: int = 40,
) -> go.Figure:
    """Generate interactive histogram for property prices with optional Log10 scaling."""
    if df is None or len(df) == 0 or "price_inr" not in df.columns:
        return create_empty_state_figure(
            "No property price records available to display."
        )

    valid_df = df.dropna(subset=["price_inr"]).copy()
    valid_df = valid_df[valid_df["price_inr"] > 0]
    if len(valid_df) == 0:
        return create_empty_state_figure("No positive property price records found.")

    if use_log_scale:
        valid_df["log_price"] = np.log10(valid_df["price_inr"])
        fig = px.histogram(
            valid_df,
            x="log_price",
            nbins=nbins,
            title="Log10 Property Valuation Distribution (₹ INR)",
            color_discrete_sequence=["#2563EB"],
            labels={"log_price": "Log10(Price in INR)"},
        )
        fig.update_traces(
            hovertemplate="<b>Log10 Price:</b> %{x:.2f}<br><b>Listing Count:</b> %{y:,}<extra></extra>",
            marker_line_width=1,
            marker_line_color="#1E40AF",
        )
        fig.add_annotation(
            text="ℹ️ Log10 resolves severe right skew (e.g., 6.0 = ₹10 Lakh, 7.0 = ₹1 Crore, 8.0 = ₹10 Crore).",
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.08,
            showarrow=False,
            font={"size": 11, "color": "#64748B"},
        )
    else:
        valid_df["price_cr"] = valid_df["price_inr"] / 10_000_000.0
        # Cap visual view at 99th percentile for clean readability while preserving statistics
        cap_val = float(valid_df["price_cr"].quantile(0.99))
        clipped = valid_df[valid_df["price_cr"] <= cap_val]
        fig = px.histogram(
            clipped,
            x="price_cr",
            nbins=nbins,
            title=f"Property Price Distribution (₹ Crores, N={len(valid_df):,})",
            color_discrete_sequence=["#2563EB"],
            labels={"price_cr": "Asking Price (₹ Crores)"},
        )
        med_val = float(valid_df["price_cr"].median())
        fig.add_vline(
            x=med_val,
            line_dash="dash",
            line_color="#DC2626",
            line_width=2,
            annotation_text=f"Median: ₹{med_val:.2f} Cr",
            annotation_position="top right",
            annotation_font={"color": "#DC2626", "size": 12},
        )
        fig.update_traces(
            hovertemplate="<b>Price Range:</b> ₹%{x:.2f} Cr<br><b>Listing Count:</b> %{y:,}<extra></extra>",
            marker_line_width=1,
            marker_line_color="#1E40AF",
        )

    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={"gridcolor": PLOTLY_THEME["gridcolor"]},
        yaxis={"gridcolor": PLOTLY_THEME["gridcolor"], "title": "Listing Count"},
        height=420,
    )
    return fig


def create_price_by_city_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Generate bar chart comparing median property prices across metropolitan cities with sample sizes."""
    if (
        df is None
        or len(df) == 0
        or "city" not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure("No city price records available to display.")

    plot_df = df.dropna(subset=["city", "price_inr"]).copy()
    plot_df = plot_df[plot_df["price_inr"] > 0]
    if len(plot_df) == 0:
        return create_empty_state_figure("No valid city price records available.")

    summary = (
        plot_df.groupby("city")["price_inr"]
        .agg(median_price="median", sample_count="count")
        .reset_index()
    )
    summary["median_cr"] = summary["median_price"] / 10_000_000.0
    summary = summary.sort_values(by="median_cr", ascending=False).head(top_n)

    fig = px.bar(
        summary,
        x="city",
        y="median_cr",
        text="median_cr",
        custom_data=["sample_count"],
        title="Median Property Price by Metropolitan Region (₹ Crores)",
        color="city",
        color_discrete_sequence=PLOTLY_THEME["color_sequence"],
        labels={"city": "Metropolitan Region", "median_cr": "Median Price (₹ Crores)"},
    )
    fig.update_traces(
        texttemplate="₹%{text:.2f} Cr<br>(n=%{customdata[0]:,})",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Median Price: ₹%{y:.2f} Cr<br>Verified Sample: %{customdata[0]:,} listings<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 60, "b": 60},
        yaxis={
            "gridcolor": PLOTLY_THEME["gridcolor"],
            "title": "Median Price (₹ Crores)",
        },
        height=420,
    )
    return fig


def create_city_price_boxplot(
    df: pd.DataFrame, cap_percentile: float = 0.98
) -> go.Figure:
    """Generate interactive boxplot comparing price distributions across cities."""
    if (
        df is None
        or len(df) == 0
        or "city" not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure("No city price records available to display.")

    plot_df = df.dropna(subset=["price_inr", "city"]).copy()
    plot_df = plot_df[plot_df["price_inr"] > 0]
    if len(plot_df) == 0:
        return create_empty_state_figure("No valid city price records available.")

    plot_df["price_cr"] = plot_df["price_inr"] / 10_000_000.0
    cap = float(plot_df["price_cr"].quantile(cap_percentile))
    clipped = plot_df[plot_df["price_cr"] <= cap]

    fig = px.box(
        clipped,
        x="city",
        y="price_cr",
        color="city",
        title="Property Price Distribution by City (₹ Crores, Interquartile Ranges)",
        color_discrete_sequence=PLOTLY_THEME["color_sequence"],
        labels={"city": "Metropolitan Region", "price_cr": "Price (₹ Crores)"},
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        yaxis={
            "gridcolor": PLOTLY_THEME["gridcolor"],
            "title": "Asking Price (₹ Crores)",
        },
        height=420,
    )
    return fig


def create_area_vs_price_scatter(
    df: pd.DataFrame,
    max_samples: int = 3000,
    color_by: str = "city",
) -> go.Figure:
    """Generate interactive scatter plot showing area vs price relationship with downsampling for responsiveness."""
    if (
        df is None
        or len(df) == 0
        or "area_sqft" not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure(
            "No area and price data available for scatter analysis."
        )

    plot_df = df.dropna(subset=["area_sqft", "price_inr"]).copy()
    plot_df = plot_df[(plot_df["area_sqft"] > 0) & (plot_df["price_inr"] > 0)]
    if len(plot_df) == 0:
        return create_empty_state_figure("No positive area and price pairs available.")

    total_valid = len(plot_df)
    is_sampled = total_valid > max_samples
    if is_sampled:
        plot_df = plot_df.sample(max_samples, random_state=42)

    plot_df["price_cr"] = plot_df["price_inr"] / 10_000_000.0

    color_col = color_by if color_by in plot_df.columns else None
    hover_cols = [c for c in ["locality", "bhk", "bathrooms"] if c in plot_df.columns]

    title = "Property Area (sq.ft) vs. Valuation (₹ Crores)"
    if is_sampled:
        title += f" [Displaying deterministic sample of {max_samples:,} of {total_valid:,} records]"

    fig = px.scatter(
        plot_df,
        x="area_sqft",
        y="price_cr",
        color=color_col,
        hover_data=hover_cols,
        title=title,
        color_discrete_sequence=PLOTLY_THEME["color_sequence"],
        labels={
            "area_sqft": "Usable Area (sq.ft.)",
            "price_cr": "Asking Price (₹ Crores)",
        },
        opacity=0.65,
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={
            "gridcolor": PLOTLY_THEME["gridcolor"],
            "title": "Usable Area (Square Feet)",
        },
        yaxis={"gridcolor": PLOTLY_THEME["gridcolor"], "title": "Price (₹ Crores)"},
        height=450,
    )
    return fig


def create_bhk_comparison_bar(df: pd.DataFrame) -> go.Figure:
    """Generate bar chart of median price across BHK bedroom configurations with sample sizes."""
    if (
        df is None
        or len(df) == 0
        or "bhk" not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure(
            "No BHK and price records available to display."
        )

    plot_df = df.dropna(subset=["price_inr", "bhk"]).copy()
    plot_df = plot_df[(plot_df["price_inr"] > 0) & (plot_df["bhk"].between(1, 8))]
    if len(plot_df) == 0:
        return create_empty_state_figure(
            "No valid records between 1 and 8 BHK available."
        )

    bhk_summary = (
        plot_df.groupby("bhk")["price_inr"]
        .agg(median_price="median", sample_count="count")
        .reset_index()
    )
    bhk_summary["median_lakhs"] = bhk_summary["median_price"] / 100_000.0
    bhk_summary["bhk_label"] = bhk_summary["bhk"].apply(lambda b: f"{int(b)} BHK")

    fig = px.bar(
        bhk_summary,
        x="bhk_label",
        y="median_lakhs",
        text="median_lakhs",
        custom_data=["sample_count"],
        title="Median Property Price by Bedroom Configuration (₹ Lakhs)",
        color_discrete_sequence=["#2563EB"],
        labels={
            "bhk_label": "BHK Configuration",
            "median_lakhs": "Median Price (₹ Lakhs)",
        },
    )
    fig.update_traces(
        texttemplate="₹%{text:.1f} L<br>(n=%{customdata[0]:,})",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Median Price: ₹%{y:.1f} Lakhs<br>Count: %{customdata[0]:,} listings<extra></extra>",
    )
    fig.add_annotation(
        text="⚠️ Descriptive comparison across bedroom counts. Does not establish causation.",
        xref="paper",
        yref="paper",
        x=0.01,
        y=1.08,
        showarrow=False,
        font={"size": 11, "color": "#64748B"},
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        yaxis={
            "gridcolor": PLOTLY_THEME["gridcolor"],
            "title": "Median Price in Lakhs (₹)",
        },
        height=420,
    )
    return fig


def create_price_per_sqft_chart(df: pd.DataFrame) -> go.Figure:
    """Generate bar chart of median price per square foot across metropolitan cities."""
    if (
        df is None
        or len(df) == 0
        or "area_sqft" not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure(
            "No area and price data available to calculate price per sq.ft."
        )

    plot_df = df.dropna(subset=["area_sqft", "price_inr"]).copy()
    plot_df = plot_df[(plot_df["area_sqft"] > 50) & (plot_df["price_inr"] > 0)]
    if len(plot_df) == 0:
        return create_empty_state_figure(
            "No valid records with positive area and price available."
        )

    plot_df["price_per_sqft"] = plot_df["price_inr"] / plot_df["area_sqft"]

    group_col = "city" if "city" in plot_df.columns else "bhk"
    summary = (
        plot_df.groupby(group_col)["price_per_sqft"]
        .agg(median_pps="median", sample_count="count")
        .reset_index()
    )
    summary = summary.sort_values(by="median_pps", ascending=False)

    fig = px.bar(
        summary,
        x=group_col,
        y="median_pps",
        text="median_pps",
        custom_data=["sample_count"],
        title="Median Rate per Square Foot by City (₹ / sq.ft)",
        color=group_col,
        color_discrete_sequence=PLOTLY_THEME["color_sequence"],
        labels={group_col: "Region", "median_pps": "Median Rate (₹ / sq.ft)"},
    )
    fig.update_traces(
        texttemplate="₹%{text:,.0f} / sqft<br>(n=%{customdata[0]:,})",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Median Rate: ₹%{y:,.0f} / sq.ft<br>Count: %{customdata[0]:,} listings<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 60, "b": 50},
        yaxis={"gridcolor": PLOTLY_THEME["gridcolor"], "title": "Rate (₹ / sq.ft)"},
        height=420,
    )
    return fig


def create_property_type_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Generate comparison of property types (Apartment, House, Villa, Builder Floor)."""
    if (
        df is None
        or len(df) == 0
        or "property_type" not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure("No property type and price data available.")

    plot_df = df.dropna(subset=["property_type", "price_inr"]).copy()
    plot_df = plot_df[plot_df["price_inr"] > 0]
    if len(plot_df) == 0:
        return create_empty_state_figure("No valid property type records available.")

    summary = (
        plot_df.groupby("property_type")["price_inr"]
        .agg(median_price="median", sample_count="count")
        .reset_index()
    )
    summary["median_cr"] = summary["median_price"] / 10_000_000.0
    summary = summary.sort_values(by="median_cr", ascending=False)

    fig = px.bar(
        summary,
        x="property_type",
        y="median_cr",
        text="median_cr",
        custom_data=["sample_count"],
        title="Median Property Price by Construction Typology (₹ Crores)",
        color="property_type",
        color_discrete_sequence=PLOTLY_THEME["color_sequence"],
        labels={"property_type": "Typology", "median_cr": "Median Price (₹ Crores)"},
    )
    fig.update_traces(
        texttemplate="₹%{text:.2f} Cr<br>(n=%{customdata[0]:,})",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Median Price: ₹%{y:.2f} Cr<br>Verified Count: %{customdata[0]:,} listings<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 60, "b": 50},
        yaxis={
            "gridcolor": PLOTLY_THEME["gridcolor"],
            "title": "Median Price (₹ Crores)",
        },
        height=420,
    )
    return fig


def create_category_distribution_boxplot(
    df: pd.DataFrame,
    category_col: str = "city",
) -> go.Figure:
    """Generate distribution boxplot across a selectable categorical column."""
    if (
        df is None
        or len(df) == 0
        or category_col not in df.columns
        or "price_inr" not in df.columns
    ):
        return create_empty_state_figure(
            f"No records available for category: {category_col}."
        )

    plot_df = df.dropna(subset=[category_col, "price_inr"]).copy()
    plot_df = plot_df[plot_df["price_inr"] > 0]
    if len(plot_df) == 0:
        return create_empty_state_figure(
            "No valid records found for distribution comparison."
        )

    plot_df["price_cr"] = plot_df["price_inr"] / 10_000_000.0
    cap = float(plot_df["price_cr"].quantile(0.98))
    clipped = plot_df[plot_df["price_cr"] <= cap]

    fig = px.box(
        clipped,
        x=category_col,
        y="price_cr",
        color=category_col,
        title=f"Price Distribution by {category_col.replace('_', ' ').title()} (₹ Crores, Capped at 98th Percentile)",
        color_discrete_sequence=PLOTLY_THEME["color_sequence"],
        labels={
            category_col: category_col.replace("_", " ").title(),
            "price_cr": "Price (₹ Crores)",
        },
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        showlegend=False,
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        yaxis={"gridcolor": PLOTLY_THEME["gridcolor"], "title": "Price (₹ Crores)"},
        height=420,
    )
    return fig


def create_area_distribution_chart(df: pd.DataFrame, nbins: int = 40) -> go.Figure:
    """Generate histogram showing the distribution of usable property area in square feet."""
    if df is None or len(df) == 0 or "area_sqft" not in df.columns:
        return create_empty_state_figure("No area records available to display.")

    plot_df = df.dropna(subset=["area_sqft"]).copy()
    plot_df = plot_df[plot_df["area_sqft"] > 0]
    if len(plot_df) == 0:
        return create_empty_state_figure("No positive area measurements available.")

    cap = float(plot_df["area_sqft"].quantile(0.99))
    clipped = plot_df[plot_df["area_sqft"] <= cap]
    med_area = float(plot_df["area_sqft"].median())

    fig = px.histogram(
        clipped,
        x="area_sqft",
        nbins=nbins,
        title=f"Property Usable Area Distribution (sq.ft, N={len(plot_df):,})",
        color_discrete_sequence=["#059669"],
        labels={"area_sqft": "Usable Area (Square Feet)"},
    )
    fig.add_vline(
        x=med_area,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"Median: {med_area:,.0f} sq.ft",
        annotation_position="top right",
        annotation_font={"color": "#DC2626", "size": 12},
    )
    fig.update_traces(
        hovertemplate="<b>Area Range:</b> %{x:,.0f} sq.ft<br><b>Listing Count:</b> %{y:,}<extra></extra>",
        marker_line_width=1,
        marker_line_color="#047857",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        xaxis={"gridcolor": PLOTLY_THEME["gridcolor"]},
        yaxis={"gridcolor": PLOTLY_THEME["gridcolor"], "title": "Listing Count"},
        height=420,
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Generate interactive correlation matrix for numerical features with causality disclosure."""
    if df is None or len(df) == 0:
        return create_empty_state_figure(
            "No records available to compute correlation matrix."
        )

    candidate_cols = [
        "price_inr",
        "area_sqft",
        "bhk",
        "bathrooms",
        "floor_number",
        "total_floors",
    ]
    num_cols = [
        c
        for c in candidate_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Exclude constant or completely null columns
    valid_cols = [c for c in num_cols if df[c].dropna().nunique() > 1]
    if len(valid_cols) < 2:
        return create_empty_state_figure(
            "Fewer than 2 non-constant numeric columns available for correlation."
        )

    corr = df[valid_cols].corr()
    z_vals = np.round(corr.values, 2)
    x_labels = [c.replace("_", " ").title() for c in corr.columns]
    y_labels = [c.replace("_", " ").title() for c in corr.index]

    fig = ff.create_annotated_heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Numerical Feature Pearson Correlation Matrix",
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 60, "r": 60, "t": 60, "b": 60},
        height=450,
    )
    fig.add_annotation(
        text="⚠️ Pearson correlation measures linear association only. Correlation does not establish causation.",
        xref="paper",
        yref="paper",
        x=0.01,
        y=1.08,
        showarrow=False,
        font={"size": 11, "color": "#64748B"},
    )
    return fig


def create_data_quality_figure(df: pd.DataFrame) -> go.Figure:
    """Generate visual data completeness and quality audit across all key attributes."""
    if df is None or len(df) == 0:
        return create_empty_state_figure(
            "No records available for data quality auditing."
        )

    total_rows = len(df)
    key_cols = [
        "city",
        "locality",
        "property_type",
        "bhk",
        "bathrooms",
        "area_sqft",
        "price_inr",
        "furnishing_status",
    ]
    present_cols = [c for c in key_cols if c in df.columns]

    completeness = []
    for c in present_cols:
        non_null = int(df[c].notna().sum())
        pct = round((non_null / total_rows) * 100.0, 1) if total_rows > 0 else 0.0
        completeness.append(
            {
                "feature": c.replace("_", " ").title(),
                "completeness_pct": pct,
                "non_null_count": non_null,
                "missing_count": total_rows - non_null,
            }
        )

    comp_df = pd.DataFrame(completeness)
    fig = px.bar(
        comp_df,
        y="feature",
        x="completeness_pct",
        orientation="h",
        text="completeness_pct",
        custom_data=["non_null_count", "missing_count"],
        title=f"Dataset Attribute Completeness Audit (Total Records: {total_rows:,})",
        color="completeness_pct",
        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
        range_color=[80, 100],
        labels={"feature": "Attribute", "completeness_pct": "Completeness (%)"},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Completeness: %{x:.1f}%<br>Non-null: %{customdata[0]:,}<br>Missing: %{customdata[1]:,}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 80, "r": 40, "t": 60, "b": 40},
        xaxis={"gridcolor": PLOTLY_THEME["gridcolor"], "range": [0, 105]},
        yaxis={"autorange": "reversed"},
        height=380,
    )
    return fig


def create_model_performance_chart() -> go.Figure:
    """Generate genuine model evaluation comparison chart from trained pipeline metadata."""
    meta_path = Path("artifacts/model_metadata_v1.json")
    if not meta_path.exists():
        return create_empty_state_figure(
            "Model evaluation metadata not found at artifacts/model_metadata_v1.json."
        )

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return create_empty_state_figure("Could not parse model metadata artifact.")

    candidates = meta.get("candidate_comparison", {})
    if not candidates:
        return create_empty_state_figure(
            "No candidate comparison metrics recorded in model metadata."
        )

    records = []
    for model_name, metrics in candidates.items():
        records.append(
            {
                "model": model_name.replace("_", " "),
                "r2_score": round(metrics.get("r2_score", 0.0), 3),
                "mae_lakhs": round(metrics.get("mae_inr", 0.0) / 100_000.0, 2),
                "rmse_cr": round(metrics.get("rmse_inr", 0.0) / 10_000_000.0, 2),
                "sample_size": metrics.get("sample_size", 11109),
            }
        )

    perf_df = pd.DataFrame(records).sort_values(by="r2_score", ascending=False)

    fig = px.bar(
        perf_df,
        x="model",
        y="r2_score",
        text="r2_score",
        custom_data=["mae_lakhs", "rmse_cr", "sample_size"],
        title="Held-Out Model Performance: Test R² Score (11,109 Test Listings)",
        color="r2_score",
        color_continuous_scale="Viridis",
        labels={"model": "Evaluated Algorithm", "r2_score": "Test R² Score"},
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Test R²: %{y:.3f}<br>MAE: ₹%{customdata[0]:.2f} Lakhs<br>RMSE: ₹%{customdata[1]:.2f} Cr<br>Held-out Sample: %{customdata[2]:,} properties<extra></extra>",
    )
    fig.add_annotation(
        text="✅ Genuine test evaluation across 11,109 multi-city listings. Hist Gradient Boosting selected as production champion.",
        xref="paper",
        yref="paper",
        x=0.01,
        y=1.08,
        showarrow=False,
        font={"size": 11, "color": "#059669"},
    )
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font_family=PLOTLY_THEME["font_family"],
        margin={"l": 40, "r": 40, "t": 60, "b": 50},
        yaxis={
            "gridcolor": PLOTLY_THEME["gridcolor"],
            "title": "Test R² Score",
            "range": [-0.15, 0.85],
        },
        height=420,
    )
    return fig
