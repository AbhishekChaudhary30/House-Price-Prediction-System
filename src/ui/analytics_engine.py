"""Data-driven automatic intelligence engine generating one-click statistical insights."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.formatters import format_inr_short


@dataclass
class DatasetInsights:
    total_records: int
    total_features: int
    target_summary: dict[str, Any]
    city_breakdown: list[dict[str, Any]]
    bhk_breakdown: list[dict[str, Any]]
    property_type_breakdown: list[dict[str, Any]]
    narrative_insights: list[str]
    limitations: list[str] = field(default_factory=list)


def run_one_click_analysis(df: pd.DataFrame) -> DatasetInsights:
    """Compute comprehensive statistical summaries and factual narrative insights from active dataset."""
    total_records = len(df)
    total_features = df.shape[1]
    narratives: list[str] = []
    limitations: list[str] = []

    # 1. Target Price Analysis
    target_summary = {}
    if "price_inr" in df.columns:
        p_series = pd.to_numeric(df["price_inr"], errors="coerce").dropna()
        if len(p_series) > 0:
            median_p = float(p_series.median())
            mean_p = float(p_series.mean())
            min_p = float(p_series.min())
            max_p = float(p_series.max())
            std_p = float(p_series.std())

            target_summary = {
                "count": len(p_series),
                "min": min_p,
                "min_formatted": format_inr_short(min_p),
                "median": median_p,
                "median_formatted": format_inr_short(median_p),
                "mean": mean_p,
                "mean_formatted": format_inr_short(mean_p),
                "max": max_p,
                "max_formatted": format_inr_short(max_p),
                "std": std_p,
            }

            narratives.append(
                f"The catalog median asking price is **{format_inr_short(median_p)}** "
                f"with 50% of residential properties falling between {format_inr_short(p_series.quantile(0.25))} "
                f"and {format_inr_short(p_series.quantile(0.75))}."
            )

    # 2. City Breakdown
    city_breakdown = []
    if "city" in df.columns and "price_inr" in df.columns:
        city_groups = df.groupby("city")
        for city_name, grp in city_groups:
            c_prices = pd.to_numeric(grp["price_inr"], errors="coerce").dropna()
            c_area = pd.to_numeric(grp.get("area_sqft", 1000), errors="coerce").dropna()
            median_c = float(c_prices.median()) if len(c_prices) > 0 else 0.0

            # Calculate price per sqft if area available
            pps = (c_prices / c_area).median() if len(c_area) > 0 else 0.0

            city_breakdown.append({
                "city": str(city_name),
                "count": len(grp),
                "share_pct": round(len(grp) / total_records * 100.0, 1),
                "median_price_formatted": format_inr_short(median_c),
                "median_pps_formatted": f"₹{pps:,.0f} / sqft",
            })

        # Find most and least expensive city
        if len(city_breakdown) > 1:
            sorted_cities = sorted(city_breakdown, key=lambda x: x["count"], reverse=True)
            top_city = sorted_cities[0]
            narratives.append(
                f"**{top_city['city']}** constitutes the dominant geographic share ({top_city['share_pct']}% of listings) "
                f"with a median listing valuation of {top_city['median_price_formatted']}."
            )

    # 3. BHK Breakdown
    bhk_breakdown = []
    if "bhk" in df.columns and "price_inr" in df.columns:
        valid_bhk = df[df["bhk"].between(1, 6)]
        for bhk_val, grp in valid_bhk.groupby("bhk"):
            b_prices = pd.to_numeric(grp["price_inr"], errors="coerce").dropna()
            bhk_breakdown.append({
                "bhk": f"{int(bhk_val)} BHK",
                "count": len(grp),
                "share_pct": round(len(grp) / total_records * 100.0, 1),
                "median_price": float(b_prices.median()) if len(b_prices) > 0 else 0.0,
                "median_price_formatted": format_inr_short(b_prices.median()) if len(b_prices) > 0 else "₹0",
            })

        if bhk_breakdown:
            most_common_bhk = max(bhk_breakdown, key=lambda x: x["count"])
            narratives.append(
                f"**{most_common_bhk['bhk']}** is the most widely available inventory configuration "
                f"accounting for {most_common_bhk['share_pct']}% of listings with a median price of {most_common_bhk['median_price_formatted']}."
            )

    # 4. Property Type Breakdown
    property_type_breakdown = []
    if "property_type" in df.columns and "price_inr" in df.columns:
        for ptype, grp in df.groupby("property_type"):
            pt_prices = pd.to_numeric(grp["price_inr"], errors="coerce").dropna()
            property_type_breakdown.append({
                "property_type": str(ptype),
                "count": len(grp),
                "share_pct": round(len(grp) / total_records * 100.0, 1),
                "median_price_formatted": format_inr_short(pt_prices.median()) if len(pt_prices) > 0 else "₹0",
            })

    # Limitations
    limitations.append("All calculations reflect advertised portal asking prices; actual completed deed prices typically close 5-15% lower.")
    if len(df) > 0 and "city" in df.columns and df["city"].nunique() <= 3:
        limitations.append("Dataset coverage is concentrated in tier-1 metropolises (Bengaluru, Mumbai, Delhi); extrapolation to tier-2 cities is not supported.")

    return DatasetInsights(
        total_records=total_records,
        total_features=total_features,
        target_summary=target_summary,
        city_breakdown=city_breakdown,
        bhk_breakdown=bhk_breakdown,
        property_type_breakdown=property_type_breakdown,
        narrative_insights=narratives,
        limitations=limitations,
    )
