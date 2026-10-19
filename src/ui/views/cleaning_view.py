"""Data cleaning view: automated standardization, before-and-after audit, and CSV export."""

import streamlit as st

from src.ui.auto_cleaner import run_automated_cleaning
from src.ui.schema_detector import detect_dataset_schema, get_active_or_default_dataset


def render_cleaning_view():
    """Render the automatic data cleaning interface, transparent audit report, and cleaned data export."""
    st.markdown("## 🧹 Automated Data Quality & Cleaning")
    st.caption("Standardize text casing, parse complex area values, eliminate duplicates, and resolve outliers without corrupting raw data.")

    raw_df = get_active_or_default_dataset()
    if raw_df is None:
        st.info("No active dataset found in memory. Please select or upload a dataset in Step 3 (Workspace).")
        if st.button("⬅️ Go to Step 3: Dataset Workspace", type="primary"):
            st.session_state["workflow_step"] = 3
            st.rerun()
        return

    schema_report = st.session_state.get("schema_report")
    if schema_report is None:
        schema_report = detect_dataset_schema(raw_df)
        st.session_state["schema_report"] = schema_report

    detected_roles = schema_report.inferred_roles if schema_report else {}

    # Action Button
    st.write("")
    clean_btn = st.button("✨ Run Automatic Data Cleaning & Standardization", type="primary", use_container_width=True)

    if clean_btn or "cleaned_df" in st.session_state:
        if clean_btn or "cleaned_df" not in st.session_state:
            with st.spinner("Executing non-destructive data cleaning pipeline..."):
                cleaned_df, report = run_automated_cleaning(raw_df, detected_roles)
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_report"] = report
                # Invalidate downstream analytics to ensure fresh recalculation
                st.session_state.pop("analytics_insights", None)

        cleaned_df = st.session_state["cleaned_df"]
        report = st.session_state["cleaning_report"]

        st.success(f"✅ Automated Cleaning Completed: Retained {report.final_row_count:,} high-quality records.")

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Initial Rows", f"{report.original_row_count:,}")
        with m2:
            st.metric("Duplicates Removed", f"{report.duplicates_removed:,}")
        with m3:
            st.metric("Outliers Filtered", f"{report.outliers_removed:,}")
        with m4:
            st.metric("Final Usable Rows", f"{report.final_row_count:,}")

        # Detailed audit transformations
        st.write("")
        st.markdown("### 📋 Cleaning Audit Log & Transformations")
        if report.modifications:
            for mod in report.modifications:
                st.markdown(f"- ✅ **Applied:** {mod}")
        else:
            st.info("No data anomalies detected; dataset conforms cleanly to baseline standards.")

        # Before and After Comparison
        st.write("")
        st.markdown("### 🔄 Before vs. After Inspection")
        tab_after, tab_before = st.tabs(["✨ Cleaned Dataset Preview", "📄 Original Raw Data Preview"])
        with tab_after:
            st.dataframe(cleaned_df.head(10))
        with tab_before:
            st.dataframe(raw_df.head(10))

        # Download Cleaned CSV
        st.write("")
        csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_housing_dataset.csv",
            mime="text/csv",
        )

    # Bottom Workflow Navigation
    st.markdown("---")
    prev_col, _, next_col = st.columns([1.5, 3, 1.5])
    with prev_col:
        if st.button("⬅️ Previous: Workspace", use_container_width=True):
            st.session_state["workflow_step"] = 3
            st.rerun()
    with next_col:
        has_cleaned = "cleaned_df" in st.session_state
        if st.button("Next: Automatic Analytics ➡️", type="primary", use_container_width=True, disabled=not has_cleaned):
            st.session_state["workflow_step"] = 5
            st.rerun()
