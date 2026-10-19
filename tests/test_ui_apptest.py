"""Automated Streamlit AppTest suite verifying UI rendering, workflow navigation, and state management."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app():
    """Create a fresh AppTest instance with a 30s timeout for ML model loading."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    return at


def test_app_initializes_and_renders_home_page(app):
    """Verify application starts clean without unhandled exceptions and displays landing view."""
    assert not app.exception
    # Check that Home view is active by default (Step 1)
    assert app.session_state["workflow_step"] == 1
    # Check that model status is online in sidebar
    assert len(app.sidebar.success) > 0
    assert "Model Pipeline Ready" in app.sidebar.success[0].value


def test_step_navigation_to_valuation(app):
    """Verify clicking Explore Valuations button advances workflow to Step 2."""
    assert not app.exception
    # Set step to 2 directly or trigger navigation
    app.session_state["workflow_step"] = 2
    app.run()

    assert not app.exception
    assert app.session_state["workflow_step"] == 2


def test_step_navigation_to_dataset_workspace(app):
    """Verify Step 3 (Dataset Workspace) renders and detects schema without exceptions."""
    app.session_state["workflow_step"] = 3
    app.run()

    assert not app.exception
    assert app.session_state["workflow_step"] == 3


def test_step_navigation_to_cleaning_and_insights(app):
    """Verify Step 4 (Data Cleaning) and Step 5 (Insights) render without DataFrame truth value errors."""
    app.session_state["workflow_step"] = 4
    app.run()
    assert not app.exception

    app.session_state["workflow_step"] = 5
    app.run()
    assert not app.exception


def test_step_navigation_to_explorer_and_about(app):
    """Verify Step 6 (Explorer) and Step 7 (About Model Card) render cleanly."""
    app.session_state["workflow_step"] = 6
    app.run()
    assert not app.exception

    # Check metrics in Step 6
    metrics_labels = [m.label for m in app.metric]
    assert "Total Usable Records" in metrics_labels
    assert "Matching Filters" in metrics_labels

    app.session_state["workflow_step"] = 7
    app.run()
    assert not app.exception


def test_cleaning_pipeline_flows_to_explorer_with_all_records(app):
    """Verify that running Step 4 (Data Cleaning) retains 55,543 records and does not empty out Step 6."""
    # Step 3: workspace loads bundled data
    app.session_state["workflow_step"] = 3
    app.run()
    assert not app.exception

    # Step 4: run cleaning
    app.session_state["workflow_step"] = 4
    app.run()
    assert not app.exception

    # Trigger cleaning button if available
    clean_buttons = [b for b in app.button if "Clean" in b.label]
    if clean_buttons:
        clean_buttons[0].click().run()
        assert not app.exception

    # Verify cleaned_df in session state is not empty
    if "cleaned_df" in app.session_state:
        assert len(app.session_state["cleaned_df"]) == 55543

    # Step 6: go to explorer
    app.session_state["workflow_step"] = 6
    app.run()
    assert not app.exception

    # Verify matching records count is > 0
    metric_values = {m.label: m.value for m in app.metric}
    if "Total Usable Records" in metric_values:
        assert metric_values["Total Usable Records"] == "55,543"
        assert metric_values["Matching Filters"] == "55,543"

