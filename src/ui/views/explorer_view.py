"""Data explorer and visualization studio view: Plotly charts, dynamic filters, and CSV export."""

import streamlit as st

from src.ui.chart_studio import (
    create_area_distribution_chart,
    create_area_vs_price_scatter,
    create_bhk_comparison_bar,
    create_category_distribution_boxplot,
    create_city_price_boxplot,
    create_correlation_heatmap,
    create_data_quality_figure,
    create_model_performance_chart,
    create_price_by_city_bar,
    create_price_distribution_chart,
    create_price_per_sqft_chart,
    create_property_type_comparison_chart,
)
from src.ui.data_access import get_validated_explorer_dataset


def render_explorer_view():
    """Render the manual data explorer and interactive visualization studio."""
    st.markdown("## 📈 Interactive Explorer & Visualization Studio")
    st.caption(
        "Slice, filter, and inspect housing micro-markets using dynamic Plotly charts and tabular querying."
    )

    # 1. Establish single reliable data source
    base_df, diag = get_validated_explorer_dataset()

    if base_df is None or len(base_df) == 0:
        st.error(
            f"⚠️ Unable to load housing dataset: {diag.error_message or 'Unknown error'}"
        )
        if st.button("⬅️ Go to Step 3: Dataset Workspace", type="primary"):
            st.session_state["workflow_step"] = 3
            st.rerun()
        return

    # 2. Extract stable candidate filter options directly from the base dataset
    cities_available = (
        sorted(base_df["city"].dropna().unique().tolist())
        if "city" in base_df.columns
        else []
    )
    bhks_available = (
        sorted([int(x) for x in base_df["bhk"].dropna().unique() if 1 <= x <= 10])
        if "bhk" in base_df.columns
        else []
    )
    prop_types_available = (
        sorted(base_df["property_type"].dropna().unique().tolist())
        if "property_type" in base_df.columns
        else []
    )
    furnish_available = (
        sorted(base_df["furnishing_status"].dropna().unique().tolist())
        if "furnishing_status" in base_df.columns
        else []
    )

    min_p_inr = (
        float(base_df["price_inr"].min())
        if "price_inr" in base_df.columns and base_df["price_inr"].notna().any()
        else 300_000.0
    )
    max_p_inr = (
        float(base_df["price_inr"].max())
        if "price_inr" in base_df.columns and base_df["price_inr"].notna().any()
        else 500_000_000.0
    )
    min_p_lakh = max(1.0, round(min_p_inr / 100_000.0, 1))
    max_p_lakh = round(max_p_inr / 100_000.0, 1)

    min_area = (
        float(base_df["area_sqft"].min())
        if "area_sqft" in base_df.columns and base_df["area_sqft"].notna().any()
        else 150.0
    )
    max_area = (
        float(base_df["area_sqft"].max())
        if "area_sqft" in base_df.columns and base_df["area_sqft"].notna().any()
        else 25000.0
    )

    # 3. Synchronize filter session state defaults
    if "exp_filter_cities" not in st.session_state:
        st.session_state["exp_filter_cities"] = cities_available
    if "exp_filter_bhks" not in st.session_state:
        st.session_state["exp_filter_bhks"] = bhks_available
    if "exp_filter_types" not in st.session_state:
        st.session_state["exp_filter_types"] = prop_types_available
    if "exp_filter_furnish" not in st.session_state:
        st.session_state["exp_filter_furnish"] = furnish_available
    if "exp_filter_locality" not in st.session_state:
        st.session_state["exp_filter_locality"] = ""
    if "exp_filter_price" not in st.session_state:
        st.session_state["exp_filter_price"] = (min_p_lakh, max_p_lakh)
    if "exp_filter_area" not in st.session_state:
        st.session_state["exp_filter_area"] = (min_area, max_area)

    # Apply filtering logic against base dataset
    filtered = base_df.copy()

    sel_cities = st.session_state["exp_filter_cities"]
    if sel_cities and "city" in filtered.columns:
        filtered = filtered[filtered["city"].isin(sel_cities)]
    elif not sel_cities and "city" in filtered.columns:
        filtered = filtered.iloc[0:0]

    sel_bhks = st.session_state["exp_filter_bhks"]
    if sel_bhks and len(filtered) > 0 and "bhk" in filtered.columns:
        filtered = filtered[filtered["bhk"].isin(sel_bhks)]
    elif not sel_bhks and "bhk" in filtered.columns:
        filtered = filtered.iloc[0:0]

    sel_types = st.session_state["exp_filter_types"]
    if sel_types and len(filtered) > 0 and "property_type" in filtered.columns:
        filtered = filtered[filtered["property_type"].isin(sel_types)]
    elif not sel_types and "property_type" in filtered.columns:
        filtered = filtered.iloc[0:0]

    sel_furnish = st.session_state["exp_filter_furnish"]
    if sel_furnish and len(filtered) > 0 and "furnishing_status" in filtered.columns:
        filtered = filtered[filtered["furnishing_status"].isin(sel_furnish)]
    elif not sel_furnish and "furnishing_status" in filtered.columns:
        filtered = filtered.iloc[0:0]

    search_query = st.session_state["exp_filter_locality"].strip()
    if search_query and len(filtered) > 0 and "locality" in filtered.columns:
        filtered = filtered[
            filtered["locality"].str.contains(search_query, case=False, na=False)
        ]

    if len(filtered) > 0 and "price_inr" in filtered.columns:
        p_low, p_high = st.session_state["exp_filter_price"]
        filtered = filtered[
            (filtered["price_inr"] >= p_low * 100_000.0)
            & (filtered["price_inr"] <= p_high * 100_000.0)
        ]

    if len(filtered) > 0 and "area_sqft" in filtered.columns:
        a_low, a_high = st.session_state["exp_filter_area"]
        filtered = filtered[
            (filtered["area_sqft"] >= a_low) & (filtered["area_sqft"] <= a_high)
        ]

    total_count = len(base_df)
    filtered_count = len(filtered)
    excluded_count = total_count - filtered_count

    # 4. Top Filter & Cohort Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Total Usable Records",
            f"{total_count:,}",
            help="Verified records in loaded project dataset",
        )
    with m2:
        st.metric(
            "Matching Filters",
            f"{filtered_count:,}",
            help="Active listings satisfying all filters",
        )
    with m3:
        st.metric(
            "Excluded by Filters",
            f"{excluded_count:,}",
            help="Listings filtered out of active query view",
        )
    with m4:
        st.caption(f"**Data Source:** `{diag.source}`")
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state["exp_filter_cities"] = cities_available
            st.session_state["exp_filter_bhks"] = bhks_available
            st.session_state["exp_filter_types"] = prop_types_available
            st.session_state["exp_filter_furnish"] = furnish_available
            st.session_state["exp_filter_locality"] = ""
            st.session_state["exp_filter_price"] = (min_p_lakh, max_p_lakh)
            st.session_state["exp_filter_area"] = (min_area, max_area)
            st.rerun()

    tab_charts, tab_table = st.tabs(
        ["📊 Interactive Visualization Studio", "🔍 Query & Filter Studio"]
    )

    # TAB 1: Visualization Studio
    with tab_charts:
        st.write("")
        c_ctrl1, c_ctrl2 = st.columns([2.5, 1.5])
        with c_ctrl1:
            chart_options = [
                "Price Distribution Histogram",
                "Price by City (Median Bar Chart)",
                "Inter-City Price Comparison (Boxplot)",
                "Area vs. Price Relationship (Scatter)",
                "Price by BHK Bedroom Configuration (Bar)",
                "Price per Square Foot by City (₹ / sq.ft)",
                "Property Typology Valuation Comparison",
                "Price Distribution by Category (Boxplot)",
                "Property Usable Area Distribution (sq.ft)",
                "Feature Correlation Heatmap",
                "Dataset Completeness & Quality Audit",
                "Candidate Model Evaluation Comparison",
            ]
            selected_chart = st.selectbox(
                "Select Visual Analytic:", chart_options, index=0
            )

        with c_ctrl2:
            scope_choice = st.radio(
                "Visual Dataset Scope:",
                [
                    f"Active Filtered ({filtered_count:,})",
                    f"Full Benchmark ({total_count:,})",
                ],
                index=0,
                horizontal=True,
            )
            plot_df = filtered if "Active" in scope_choice else base_df

        st.markdown("---")

        if selected_chart == "Price Distribution Histogram":
            c_opt1, c_opt2 = st.columns([1.5, 2.5])
            with c_opt1:
                use_log = st.checkbox(
                    "Apply Log10 Transformation (Resolves Right Skew)", value=False
                )
            with c_opt2:
                bins = st.slider(
                    "Histogram Bins", min_value=10, max_value=100, value=40, step=5
                )
            fig = create_price_distribution_chart(
                plot_df, use_log_scale=use_log, nbins=bins
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Price by City (Median Bar Chart)":
            fig = create_price_by_city_bar(plot_df, top_n=10)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Inter-City Price Comparison (Boxplot)":
            fig = create_city_price_boxplot(plot_df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Area vs. Price Relationship (Scatter)":
            color_opt = st.selectbox(
                "Color Scatter By:",
                [c for c in ["city", "bhk", "property_type"] if c in plot_df.columns],
                index=0,
            )
            fig = create_area_vs_price_scatter(
                plot_df, max_samples=3000, color_by=color_opt
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Price by BHK Bedroom Configuration (Bar)":
            fig = create_bhk_comparison_bar(plot_df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Price per Square Foot by City (₹ / sq.ft)":
            fig = create_price_per_sqft_chart(plot_df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Property Typology Valuation Comparison":
            fig = create_property_type_comparison_chart(plot_df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Price Distribution by Category (Boxplot)":
            cat_choice = st.selectbox(
                "Distribution Category:",
                [c for c in ["city", "bhk", "property_type"] if c in plot_df.columns],
                index=0,
            )
            fig = create_category_distribution_boxplot(plot_df, category_col=cat_choice)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Property Usable Area Distribution (sq.ft)":
            area_bins = st.slider(
                "Area Histogram Bins", min_value=10, max_value=100, value=40, step=5
            )
            fig = create_area_distribution_chart(plot_df, nbins=area_bins)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Feature Correlation Heatmap":
            fig = create_correlation_heatmap(plot_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Insufficient numeric variables present in active dataset to compute correlation matrix."
                )

        elif selected_chart == "Dataset Completeness & Quality Audit":
            fig = create_data_quality_figure(plot_df)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Candidate Model Evaluation Comparison":
            fig = create_model_performance_chart()
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Query & Filter Studio
    with tab_table:
        st.write("")
        st.markdown("#### Dynamic Real Estate Filter Controls")

        # Row 1: Categorical Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if cities_available:
                chosen_cities = st.multiselect(
                    "Filter by Metropolitan Region (City)",
                    options=cities_available,
                    default=st.session_state["exp_filter_cities"],
                    key="ui_filter_cities",
                )
                if chosen_cities != st.session_state["exp_filter_cities"]:
                    st.session_state["exp_filter_cities"] = chosen_cities
                    st.rerun()

        with fc2:
            if bhks_available:
                chosen_bhk = st.multiselect(
                    "Filter by Bedroom Count (BHK)",
                    options=bhks_available,
                    default=st.session_state["exp_filter_bhks"],
                    key="ui_filter_bhks",
                )
                if chosen_bhk != st.session_state["exp_filter_bhks"]:
                    st.session_state["exp_filter_bhks"] = chosen_bhk
                    st.rerun()

        with fc3:
            if prop_types_available:
                chosen_types = st.multiselect(
                    "Filter by Property Type",
                    options=prop_types_available,
                    default=st.session_state["exp_filter_types"],
                    key="ui_filter_types",
                )
                if chosen_types != st.session_state["exp_filter_types"]:
                    st.session_state["exp_filter_types"] = chosen_types
                    st.rerun()

        # Row 2: Secondary Filters & Range Sliders
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if furnish_available:
                chosen_furnish = st.multiselect(
                    "Filter by Furnishing Status",
                    options=furnish_available,
                    default=st.session_state["exp_filter_furnish"],
                    key="ui_filter_furnish",
                )
                if chosen_furnish != st.session_state["exp_filter_furnish"]:
                    st.session_state["exp_filter_furnish"] = chosen_furnish
                    st.rerun()

        with sc2:
            search_query = st.text_input(
                "Search Locality / Neighborhood",
                value=st.session_state["exp_filter_locality"],
                placeholder="e.g. Whitefield, Bandra, Rohini...",
                key="ui_filter_locality",
            )
            if search_query != st.session_state["exp_filter_locality"]:
                st.session_state["exp_filter_locality"] = search_query
                st.rerun()

        with sc3:
            price_slider = st.slider(
                "Price Range (₹ Lakhs)",
                min_value=float(min_p_lakh),
                max_value=float(max_p_lakh),
                value=st.session_state["exp_filter_price"],
                step=5.0,
                key="ui_filter_price",
            )
            if price_slider != st.session_state["exp_filter_price"]:
                st.session_state["exp_filter_price"] = price_slider
                st.rerun()

        st.write("")
        st.markdown(
            f"**Showing {filtered_count:,} matching records** (out of {total_count:,} total valid records)"
        )

        # Table Display
        cols_to_show = [
            c
            for c in [
                "city",
                "locality",
                "property_type",
                "bhk",
                "bathrooms",
                "area_sqft",
                "price_inr",
                "furnishing_status",
            ]
            if c in filtered.columns
        ]

        if filtered_count > 0:
            st.dataframe(
                filtered[cols_to_show].head(100),
                height=380,
                use_container_width=True,
            )
            if filtered_count > 100:
                st.caption(
                    f"Showing first 100 of {filtered_count:,} matching listings. Use CSV export below to download all {filtered_count:,} filtered records."
                )

            csv_filtered = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download Filtered Records ({filtered_count:,} listings CSV)",
                data=csv_filtered,
                file_name="filtered_housing_records.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.info(
                "ℹ️ No property listings match the selected filter criteria. Adjust your filters or click **🔄 Reset All Filters** above."
            )
            st.button("📥 Download Filtered Results (0 records)", disabled=True)

    # Bottom Workflow Navigation
    st.markdown("---")
    prev_col, _, next_col = st.columns([1.5, 3, 1.5])
    with prev_col:
        if st.button("⬅️ Previous: Analytics", use_container_width=True):
            st.session_state["workflow_step"] = 5
            st.rerun()
    with next_col:
        if st.button(
            "Next: Project Information ➡️", type="primary", use_container_width=True
        ):
            st.session_state["workflow_step"] = 7
            st.rerun()
