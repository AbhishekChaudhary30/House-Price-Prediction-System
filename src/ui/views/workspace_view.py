"""Dataset workspace view: data ingestion, CSV upload, and automated schema detection."""

import io

import pandas as pd
import streamlit as st

from src.data.load_data import load_processed_data
from src.ui.schema_detector import detect_dataset_schema


def render_workspace_view():
    """Render the dataset workspace interface for loading, uploading, and inspecting housing schemas."""
    st.markdown("## 📂 Dataset Workspace & Schema Detection")
    st.caption("Upload custom housing data or explore the bundled multi-city benchmark dataset (55,543 records).")

    source_choice = st.radio(
        "Select Dataset Source:",
        ["📦 Bundled Multi-City Dataset (55,543 verified listings)", "📤 Upload Custom Real Estate CSV"],
        index=0,
        horizontal=True,
    )

    if source_choice.startswith("📦"):
        if "active_df" not in st.session_state or st.session_state.get("dataset_source") != "bundled":
            with st.spinner("Loading bundled benchmark dataset..."):
                try:
                    df = load_processed_data()
                    st.session_state["active_df"] = df
                    st.session_state["dataset_source"] = "bundled"
                    st.session_state["schema_report"] = detect_dataset_schema(df)
                    # Clear downstream cleaning cache for consistency
                    st.session_state.pop("cleaned_df", None)
                    st.session_state.pop("cleaning_report", None)
                    st.session_state.pop("analytics_insights", None)
                except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
                    st.error(f"Error loading bundled dataset: {e}")
                    return
    else:
        uploaded_file = st.file_uploader(
            "Upload Real Estate CSV File",
            type=["csv"],
            help="Supports CSV files with standard delimiters, UTF-8/Latin-1 encodings.",
        )

        if uploaded_file is not None and st.session_state.get("uploaded_filename") != uploaded_file.name:
            try:
                bytes_data = uploaded_file.read()
                # Try UTF-8 then fallback to Latin-1
                try:
                    df = pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1")

                st.session_state["active_df"] = df
                st.session_state["dataset_source"] = "uploaded"
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.session_state["schema_report"] = detect_dataset_schema(df)
                # Invalidate stale downstream results
                st.session_state.pop("cleaned_df", None)
                st.session_state.pop("cleaning_report", None)
                st.session_state.pop("analytics_insights", None)
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df):,} rows).")
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, UnicodeDecodeError) as e:
                st.error(f"Failed to parse uploaded CSV: {e}")
                return

    # Schema Detection & Data Inspection
    if "active_df" in st.session_state:
        df = st.session_state["active_df"]
        report = st.session_state["schema_report"]

        st.write("")
        st.markdown("### 🔍 Inferred Semantic Schema & Profiling")

        # Top summary cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", f"{report.total_rows:,}")
        with c2:
            st.metric("Total Columns", report.total_columns)
        with c3:
            st.metric("Duplicate Rows", f"{report.duplicate_rows:,}")
        with c4:
            badge = "🟢 Compatible" if report.supports_valuation else "🟡 Partial / Generic"
            st.metric("Model Compatibility", badge)

        if not report.supports_valuation:
            st.warning(
                f"Note: This dataset is missing the following fields for full ML property valuation: "
                f"{', '.join(report.missing_valuation_requirements)}. "
                "You can still inspect, clean, and visualize the data in subsequent steps!"
            )

        # Tabular Column Profile
        profiles_data = []
        for col_name, prof in report.column_profiles.items():
            profiles_data.append({
                "Column Name": col_name,
                "Inferred Role": prof.inferred_role.upper(),
                "Data Type": prof.dtype,
                "Missing Values": f"{prof.missing_count} ({prof.missing_percentage}%)",
                "Unique Values": prof.unique_count,
                "Sample Values": str(prof.sample_values[:2]),
            })

        st.dataframe(pd.DataFrame(profiles_data), height=260)

        # Data Preview
        with st.expander("👀 Preview First 10 Records", expanded=False):
            st.dataframe(df.head(10))

    # Bottom Workflow Navigation
    st.markdown("---")
    prev_col, _, next_col = st.columns([1.5, 3, 1.5])
    with prev_col:
        if st.button("⬅️ Previous: Valuation", use_container_width=True):
            st.session_state["workflow_step"] = 2
            st.rerun()
    with next_col:
        has_data = "active_df" in st.session_state
        if st.button("Next: Data Cleaning ➡️", type="primary", use_container_width=True, disabled=not has_data):
            st.session_state["workflow_step"] = 4
            st.rerun()
