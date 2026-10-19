"""Automatic insights view: one-click statistical intelligence and narrative summaries."""

import json

import pandas as pd
import streamlit as st

from src.ui.analytics_engine import run_one_click_analysis
from src.ui.schema_detector import get_active_or_default_dataset


def render_insights_view():
    """Render the one-click automatic analytics and statistical narrative engine."""
    st.markdown("## 📊 Automatic Intelligence & One-Click Analytics")
    st.caption("Generate factual, data-driven real-estate insights without configuring manual formulas or charts.")

    # Determine which dataset to analyze (prefer cleaned, fallback to active/bundled)
    target_df = get_active_or_default_dataset()

    if target_df is None:
        st.info("No housing dataset found in memory. Please select a dataset in Step 3 (Workspace).")
        if st.button("⬅️ Go to Step 3: Dataset Workspace", type="primary"):
            st.session_state["workflow_step"] = 3
            st.rerun()
        return

    # One-Click Analysis Button
    st.write("")
    analyze_btn = st.button("🚀 Run Automatic Statistical Analysis", type="primary", use_container_width=True)

    if analyze_btn or "analytics_insights" in st.session_state:
        if analyze_btn or "analytics_insights" not in st.session_state:
            with st.spinner("Analyzing dataset distributions and computing cross-city intelligence..."):
                insights = run_one_click_analysis(target_df)
                st.session_state["analytics_insights"] = insights

        insights = st.session_state["analytics_insights"]

        # 1. Headline Statistics
        st.write("")
        st.markdown("### 📌 Key Market Metrics")
        ts = insights.target_summary
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Listings Analyzed", f"{insights.total_records:,}")
        with m2:
            st.metric("Median Asking Valuation", ts.get("median_formatted", "N/A"))
        with m3:
            st.metric("Minimum Asking Price", ts.get("min_formatted", "N/A"))
        with m4:
            st.metric("Maximum Asking Price", ts.get("max_formatted", "N/A"))

        # 2. Narrative Insights
        st.write("")
        st.markdown("### 💡 Factual Narrative Findings")
        for statement in insights.narrative_insights:
            st.markdown(
                f"""
                <div class="premium-card" style="padding: 1rem 1.25rem; margin-bottom: 0.75rem; border-left: 4px solid #2563EB;">
                    <span style="font-size: 1.1rem; margin-right: 0.5rem;">🔎</span>
                    <span>{statement}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Inter-City Price & Density Comparison Table
        if insights.city_breakdown:
            st.write("")
            st.markdown("### 🏙️ Metropolitan Pricing Comparison")
            city_df = pd.DataFrame(insights.city_breakdown)
            city_df.columns = ["City", "Listing Count", "Share (%)", "Median Valuation", "Median Rate / Sq.Ft."]
            st.dataframe(city_df, hide_index=True)

        # 4. BHK Inventory Breakdown Table
        if insights.bhk_breakdown:
            st.write("")
            st.markdown("### 🛏️ Inventory & Price by BHK Configuration")
            bhk_df = pd.DataFrame(insights.bhk_breakdown)
            bhk_df = bhk_df[["bhk", "count", "share_pct", "median_price_formatted"]]
            bhk_df.columns = ["Configuration", "Listing Count", "Market Share (%)", "Median Valuation"]
            st.dataframe(bhk_df, hide_index=True)

        # 5. Methodological Disclaimers
        st.write("")
        with st.expander("ℹ️ Analytical Limitations & Disclaimers", expanded=False):
            for lim in insights.limitations:
                st.markdown(f"- {lim}")

        # Download Analysis Report
        report_payload = json.dumps({
            "total_records": insights.total_records,
            "target_summary": insights.target_summary,
            "city_breakdown": insights.city_breakdown,
            "bhk_breakdown": insights.bhk_breakdown,
            "narrative_insights": insights.narrative_insights,
            "limitations": insights.limitations,
        }, indent=2)

        st.download_button(
            label="📥 Download Analytics Summary Report (JSON)",
            data=report_payload,
            file_name="automatic_analytics_report.json",
            mime="application/json",
        )

    # Bottom Workflow Navigation
    st.markdown("---")
    prev_col, _, next_col = st.columns([1.5, 3, 1.5])
    with prev_col:
        if st.button("⬅️ Previous: Cleaning", use_container_width=True):
            st.session_state["workflow_step"] = 4
            st.rerun()
    with next_col:
        if st.button("Next: Visualizations ➡️", type="primary", use_container_width=True):
            st.session_state["workflow_step"] = 6
            st.rerun()
