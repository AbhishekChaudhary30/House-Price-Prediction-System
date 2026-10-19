"""India House Price Prediction System - Premium UI/UX & Interactive Workflow Application."""

import streamlit as st

from src.services.prediction_service import HousingPredictionService
from src.ui.styles import apply_custom_styles, render_step_progress_bar
from src.ui.views.about_view import render_about_view
from src.ui.views.cleaning_view import render_cleaning_view
from src.ui.views.explorer_view import render_explorer_view
from src.ui.views.home_view import render_home_view
from src.ui.views.insights_view import render_insights_view
from src.ui.views.valuation_view import render_valuation_view
from src.ui.views.workspace_view import render_workspace_view

st.set_page_config(
    page_title="India House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading trained machine learning pipeline...")
def load_cached_service():
    """Load and cache the prediction service and serialized ML model."""
    try:
        return HousingPredictionService()
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        return None, str(e)


def main():
    # 1. Apply Ultra-Premium Light Theme Styling
    apply_custom_styles()

    # 2. Service Initialization
    service_res = load_cached_service()
    if isinstance(service_res, tuple):
        service, err = service_res
    else:
        service, err = service_res, None

    # 3. Session State Workflow Initialization
    if "workflow_step" not in st.session_state:
        st.session_state["workflow_step"] = 1

    steps_map = {
        1: "1. 🏠 Home Overview",
        2: "2. 🔮 Valuation Engine",
        3: "3. 📂 Dataset Workspace",
        4: "4. 🧹 Data Cleaning",
        5: "5. 💡 Automated Analytics",
        6: "6. 📈 Visual Explorer",
        7: "7. ℹ️ Project & Model Card",
    }

    # 4. Sidebar Controls & System Status
    with st.sidebar:
        st.markdown("### 🧭 Navigation Workflow")
        current_step_name = steps_map[st.session_state["workflow_step"]]
        selected_nav = st.radio(
            "Go to Step:",
            options=list(steps_map.values()),
            index=list(steps_map.values()).index(current_step_name),
            label_visibility="collapsed",
        )

        # Synchronize radio click with session step
        for num, label in steps_map.items():
            if label == selected_nav and num != st.session_state["workflow_step"]:
                st.session_state["workflow_step"] = num
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        if service is not None:
            meta = service.metadata
            st.success("🟢 Model Pipeline Ready")
            st.caption(f"**Champion:** `{meta.get('model_name', 'Hist_Gradient_Boosting')}`")
            st.caption(f"**Test R²:** `{meta.get('test_metrics', {}).get('r2_score', 0.7108):.4f}`")
            st.caption(f"**Indexed Records:** `{meta.get('train_samples', 44434) + meta.get('test_samples', 11109):,}`")
        else:
            st.error("🔴 Model Artifact Missing")
            if err:
                st.caption(f"Detail: {err}")

        st.markdown("---")
        if st.button("🔄 Reset Workflow", use_container_width=True):
            st.session_state["workflow_step"] = 1
            st.session_state.pop("last_prediction", None)
            st.session_state.pop("cleaned_df", None)
            st.session_state.pop("cleaning_report", None)
            st.session_state.pop("analytics_insights", None)
            st.rerun()

    # 5. Top Step Progress Bar
    render_step_progress_bar(current_step=st.session_state["workflow_step"], total_steps=7)

    # 6. Active Step View Rendering
    current_step = st.session_state["workflow_step"]
    if current_step == 1:
        render_home_view(service)
    elif current_step == 2:
        render_valuation_view(service)
    elif current_step == 3:
        render_workspace_view()
    elif current_step == 4:
        render_cleaning_view()
    elif current_step == 5:
        render_insights_view()
    elif current_step == 6:
        render_explorer_view()
    elif current_step == 7:
        render_about_view(service)


if __name__ == "__main__":
    main()
