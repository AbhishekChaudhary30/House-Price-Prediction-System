"""Ultra-premium light interface design system and custom CSS injection for Streamlit."""

import streamlit as st

CUSTOM_CSS = """
<style>
/* ----------------------------------------------------
   Ultra-Premium Light Design System Tokens
---------------------------------------------------- */
:root {
    --bg-main: #F8FAFC;
    --bg-card: #FFFFFF;
    --border-subtle: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #64748B;
    --accent-primary: #2563EB;
    --accent-hover: #1D4ED8;
    --accent-light: #EFF6FF;
    --success-green: #059669;
    --success-light: #ECFDF5;
    --warning-amber: #D97706;
    --warning-light: #FFFBEB;
    --purple-accent: #7C3AED;
    --purple-light: #F5F3FF;
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    --radius-lg: 12px;
    --radius-md: 8px;
}

/* Base Body Styling */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Custom Card Container */
.premium-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease-in-out;
}
.premium-card:hover {
    box-shadow: var(--shadow-md);
    border-color: #CBD5E1;
}

/* Highlight / Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
    color: #FFFFFF !important;
    border-radius: var(--radius-lg);
    padding: 2.25rem 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md);
}
.hero-banner h1, .hero-banner h2, .hero-banner h3, .hero-banner p {
    color: #FFFFFF !important;
}

/* Metric Display Cards */
.metric-box {
    background: #FFFFFF;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.metric-box .metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.metric-box .metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-box .metric-sub {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* Badge Tags */
.badge-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-blue { background: var(--accent-light); color: var(--accent-primary); }
.badge-green { background: var(--success-light); color: var(--success-green); }
.badge-amber { background: var(--warning-light); color: var(--warning-amber); }
.badge-purple { background: var(--purple-light); color: var(--purple-accent); }

/* Step Navigation Bar */
.workflow-nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.75rem;
    box-shadow: var(--shadow-sm);
}
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-muted);
}
.step-indicator.active {
    color: var(--accent-primary);
    font-weight: 700;
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #F1F5F9;
    color: var(--text-secondary);
    font-size: 0.75rem;
}
.step-indicator.active .step-number {
    background: var(--accent-primary);
    color: #FFFFFF;
}

/* Bottom Action Bar */
.workflow-action-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 1.5rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border-subtle);
}
</style>
"""


def apply_custom_styles():
    """Inject premium CSS styling into Streamlit document head."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_step_progress_bar(current_step: int, total_steps: int = 7):
    """Render horizontal step indicator bar showing workflow progress."""
    step_titles = [
        "Home",
        "Valuation",
        "Workspace",
        "Data Cleaning",
        "Analytics",
        "Explorer",
        "About",
    ]

    cols = st.columns(total_steps)
    for idx, (col, title) in enumerate(zip(cols, step_titles, strict=True), start=1):
        with col:
            is_active = idx == current_step
            is_done = idx < current_step
            badge_color = "#2563EB" if is_active else ("#059669" if is_done else "#94A3B8")
            bg_color = "#EFF6FF" if is_active else ("#ECFDF5" if is_done else "#F8FAFC")
            border_style = f"2px solid {badge_color}" if is_active else "1px solid #E2E8F0"
            icon = "✓" if is_done else str(idx)

            st.markdown(
                f"""
                <div style="background:{bg_color}; border:{border_style}; border-radius:8px; padding:8px 6px; text-align:center;">
                    <div style="font-size:11px; font-weight:700; color:{badge_color};">Step {icon}</div>
                    <div style="font-size:12px; font-weight:{'700' if is_active else '500'}; color:#0F172A; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")
