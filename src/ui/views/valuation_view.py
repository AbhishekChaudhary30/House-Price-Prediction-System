import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.config.settings import (
    PRIMARY_SUPPORTED_CITIES,
    VALID_FURNISHING_STATUSES,
    VALID_PROPERTY_TYPES,
)
from src.services.prediction_service import PredictionRequest
from src.utils.formatters import format_inr_short


def render_valuation_view(service):
    """Render the interactive property price estimation interface and scenario simulator."""
    st.markdown("## 🔮 Residential Property Valuation Engine")
    st.caption("Estimate fair market asking prices using the champion HistGradientBoosting model trained on 55,500+ listings.")

    if service is None:
        st.error("Trained machine learning pipeline is currently unavailable. Please run model training first.")
        return

    loc_registry = service.location_registry.get("cities", {})

    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Location & Property Classification")
        c1, c2, c3 = st.columns(3)

        with c1:
            city_options = PRIMARY_SUPPORTED_CITIES + ["Other City (Test Limitation)"]
            selected_city = st.selectbox(
                "Metropolitan City",
                city_options,
                index=0,
                help="Valuation models are currently trained for Bengaluru, Mumbai, and Delhi.",
            )

        with c2:
            city_data = loc_registry.get(selected_city, {})
            loc_dict = city_data.get("localities", {})

            if loc_dict:
                # Format locality labels with observation counts
                loc_labels = [f"{loc} ({meta['observation_count']} listings)" for loc, meta in sorted(loc_dict.items())]
                selected_label = st.selectbox("Locality / Micro-Market", loc_labels, index=0)
                selected_locality = selected_label.split(" (")[0]
            else:
                selected_locality = st.text_input("Locality / Neighborhood", value="Whitefield")

        with c3:
            prop_type = st.selectbox("Property Type", VALID_PROPERTY_TYPES, index=0)

        st.markdown("---")
        st.markdown("#### 2. Space, Elevation & Interior Configuration")
        col_dim1, col_dim2, col_dim3 = st.columns(3)

        with col_dim1:
            area_val = st.number_input("Property Area", min_value=150.0, max_value=25000.0, value=1250.0, step=50.0)
            area_unit = st.selectbox("Area Unit", ["sqft", "sq_yards", "sq_meters"], index=0)

        with col_dim2:
            bhk = st.slider("Bedrooms (BHK)", min_value=1, max_value=8, value=2)
            bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2)

        with col_dim3:
            furnishing = st.selectbox("Furnishing Status", VALID_FURNISHING_STATUSES, index=1)
            floor_num = st.number_input("Floor Number (0 = Ground)", min_value=0, max_value=100, value=3, step=1)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=120, value=8, step=1)

            if floor_num > total_floors:
                st.error("⚠️ Floor number cannot exceed total building floors.")

        st.markdown("---")
        st.markdown("#### 3. Amenities")
        am_cols = st.columns(3)
        with am_cols[0]:
            parking = st.checkbox("Dedicated Parking", value=True)
        with am_cols[1]:
            lift = st.checkbox("Elevator / Lift Access", value=True)
        with am_cols[2]:
            gated = st.checkbox("Gated Society / Security", value=True)

        st.write("")
        btn_c1, btn_c2 = st.columns([3, 1])
        with btn_c1:
            calculate_btn = st.button("🔮 Calculate Estimated Market Valuation", type="primary", use_container_width=True)
        with btn_c2:
            reset_val_btn = st.button("🔄 New Valuation", use_container_width=True)
            if reset_val_btn:
                st.session_state.pop("last_prediction", None)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Inference Execution
    if calculate_btn:
        req = PredictionRequest(
            city=selected_city,
            locality=selected_locality,
            area_value=area_val,
            area_unit=area_unit,
            bhk=int(bhk),
            bathrooms=int(bathrooms),
            property_type=prop_type,
            furnishing_status=furnishing,
            floor_number=int(floor_num),
            total_floors=int(total_floors),
            parking_available=parking,
            lift_available=lift,
            gated_community=gated,
        )

        with st.spinner("Generating valuation via Champion ML Pipeline..."):
            resp = service.predict(req)
            st.session_state["last_prediction"] = resp

            if resp.success:
                if "prediction_history" not in st.session_state:
                    st.session_state["prediction_history"] = []

                st.session_state["prediction_history"].append(
                    {
                        "Timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                        "City": resp.city,
                        "Locality": resp.locality,
                        "Type": prop_type,
                        "BHK": int(bhk),
                        "Area (sq.ft)": f"{resp.normalized_area_sqft:,.0f}",
                        "Estimated Price": resp.formatted_short,
                        "Rate / sq.ft": resp.formatted_price_per_sqft,
                        "Confidence": resp.confidence_tier,
                    }
                )

    # Display Valuation Results
    if "last_prediction" in st.session_state:
        resp = st.session_state["last_prediction"]
        if not resp.success:
            st.error("⚠️ Valuation Unsuccessful:")
            for err in resp.validation_errors:
                st.write(f"- {err}")
            if resp.limitations:
                st.info("System Limitation Context:\n- " + "\n- ".join(resp.limitations))
        else:
            st.markdown("### 🏆 Estimated Valuation Summary")
            r1, r2, r3 = st.columns([1.5, 1, 1])

            with r1:
                st.markdown(
                    f"""
                    <div class="metric-box" style="border-left: 4px solid #2563EB;">
                        <div class="metric-label">Estimated Market Asking Price</div>
                        <div class="metric-value" style="color:#1E40AF;">{resp.formatted_short}</div>
                        <div class="metric-sub" style="font-weight:600;">{resp.formatted_price}</div>
                        <div style="font-size:0.8rem; color:#475569; margin-top:0.4rem;">
                            Empirical Range: <b>{resp.formatted_interval}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Rate per Sq.Ft.</div>
                        <div class="metric-value" style="color:#059669;">{resp.formatted_price_per_sqft}</div>
                        <div class="metric-sub">Normalized: {resp.normalized_area_sqft:.0f} sq.ft.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r3:
                badge_class = "badge-green" if resp.confidence_tier == "HIGH" else "badge-amber"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Confidence & Support</div>
                        <div style="margin: 0.5rem 0;"><span class="badge-pill {badge_class}">{resp.confidence_tier} CONFIDENCE</span></div>
                        <div class="metric-sub">Micro-market: {resp.locality_observations} listings</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Scenario Simulator
            st.write("")
            st.markdown("#### ⚡ Interactive What-If Scenario Simulator")
            st.caption("Inspect how configuration adjustments impact the estimated property valuation:")

            sim_c1, sim_c2, sim_c3 = st.columns(3)

            # Scenario A: Extra 250 sqft
            req_sim_area = PredictionRequest(
                city=resp.city, locality=resp.locality, area_value=resp.normalized_area_sqft + 250,
                bhk=bhk, bathrooms=bathrooms, property_type=prop_type, furnishing_status=furnishing,
                floor_number=floor_num, total_floors=total_floors, parking_available=parking,
            )
            resp_area = service.predict(req_sim_area)
            diff_area = resp_area.estimated_price_inr - resp.estimated_price_inr

            # Scenario B: Upgrading Furnishing to Furnished
            req_sim_furn = PredictionRequest(
                city=resp.city, locality=resp.locality, area_value=resp.normalized_area_sqft,
                bhk=bhk, bathrooms=bathrooms, property_type=prop_type, furnishing_status="Furnished",
                floor_number=floor_num, total_floors=total_floors, parking_available=parking,
            )
            resp_furn = service.predict(req_sim_furn)
            diff_furn = resp_furn.estimated_price_inr - resp.estimated_price_inr

            # Scenario C: Adding Dedicated Parking
            req_sim_park = PredictionRequest(
                city=resp.city, locality=resp.locality, area_value=resp.normalized_area_sqft,
                bhk=bhk, bathrooms=bathrooms, property_type=prop_type, furnishing_status=furnishing,
                floor_number=floor_num, total_floors=total_floors, parking_available=True,
            )
            resp_park = service.predict(req_sim_park)
            diff_park = resp_park.estimated_price_inr - resp.estimated_price_inr

            with sim_c1:
                st.markdown(
                    f"""
                    <div class="premium-card">
                        <div style="font-weight:700; color:#0F172A;">+250 Sq.Ft. Area</div>
                        <div style="font-size:1.25rem; font-weight:700; color:#2563EB;">{resp_area.formatted_short}</div>
                        <div style="font-size:0.85rem; color:#059669;">+ {format_inr_short(diff_area)} increase</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with sim_c2:
                sign = "+" if diff_furn >= 0 else "-"
                st.markdown(
                    f"""
                    <div class="premium-card">
                        <div style="font-weight:700; color:#0F172A;">Full Furnishing</div>
                        <div style="font-size:1.25rem; font-weight:700; color:#2563EB;">{resp_furn.formatted_short}</div>
                        <div style="font-size:0.85rem; color:#059669;">{sign} {format_inr_short(abs(diff_furn))} delta</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with sim_c3:
                sign = "+" if diff_park >= 0 else "-"
                st.markdown(
                    f"""
                    <div class="premium-card">
                        <div style="font-weight:700; color:#0F172A;">Dedicated Parking</div>
                        <div style="font-size:1.25rem; font-weight:700; color:#2563EB;">{resp_park.formatted_short}</div>
                        <div style="font-size:0.85rem; color:#059669;">{sign} {format_inr_short(abs(diff_park))} delta</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # JSON Download
            download_payload = json.dumps({
                "city": resp.city,
                "locality": resp.locality,
                "area_sqft": resp.normalized_area_sqft,
                "estimated_price_inr": resp.estimated_price_inr,
                "formatted_price": resp.formatted_price,
                "price_per_sqft": resp.price_per_sqft_inr,
                "confidence_tier": resp.confidence_tier,
                "empirical_range": resp.formatted_interval,
                "model_version": resp.model_version,
            }, indent=2)

            st.download_button(
                label="📥 Download Valuation Summary (JSON)",
                data=download_payload,
                file_name=f"valuation_{resp.city}_{resp.locality}.json",
                mime="application/json",
            )

    # Session Valuation History & Comparison
    if st.session_state.get("prediction_history"):
        st.write("")
        st.markdown("### 📜 Session Valuation History & Comparison")
        st.caption("Review and compare all property valuations computed during this active session:")
        hist_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(hist_df, hide_index=True)

        h_col1, h_col2, _ = st.columns([1.5, 1.5, 3])
        with h_col1:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History (CSV)",
                data=csv_data,
                file_name="valuation_session_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with h_col2:
            if st.button("🗑️ Clear Valuation History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.rerun()

    # Bottom Workflow Navigation
    st.markdown("---")
    prev_col, _, next_col = st.columns([1.5, 3, 1.5])
    with prev_col:
        if st.button("⬅️ Previous: Home", use_container_width=True):
            st.session_state["workflow_step"] = 1
            st.rerun()
    with next_col:
        if st.button("Next: Workspace ➡️", type="primary", use_container_width=True):
            st.session_state["workflow_step"] = 3
            st.rerun()
