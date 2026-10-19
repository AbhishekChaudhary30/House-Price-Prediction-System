"""Formatting utilities for Indian currency, area units, and location strings."""

import re


def format_inr(amount: float) -> str:
    """Format an amount in standard Indian numbering system (e.g. ₹1,25,000 or ₹1,50,00,000)."""
    if amount is None or amount < 0:
        return "₹0"

    amount_int = round(amount)
    s = str(amount_int)
    if len(s) <= 3:
        return f"₹{s}"

    # Last 3 digits
    last_three = s[-3:]
    remaining = s[:-3]

    # Group the remaining digits in pairs of 2 from right to left
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)

    formatted_num = ",".join(groups) + "," + last_three
    return f"₹{formatted_num}"


def format_inr_short(amount: float, decimals: int = 2) -> str:
    """Format an amount into concise Indian terms (Thousands, Lakhs, Crores).

    Examples:
        75,000 -> ₹75 K
        3,500,000 -> ₹35.00 L (or ₹35 Lakh)
        12,500,000 -> ₹1.25 Cr (or ₹1.25 Crore)
    """
    if amount is None or amount <= 0:
        return "₹0"

    if amount >= 10_000_000:  # 1 Crore = 10,000,000
        cr = amount / 10_000_000
        return f"₹{cr:.{decimals}f} Cr"
    elif amount >= 100_000:  # 1 Lakh = 100,000
        lakh = amount / 100_000
        return f"₹{lakh:.{decimals}f} Lakh"
    elif amount >= 1_000:  # 1 Thousand
        k = amount / 1_000
        return f"₹{k:.{decimals}f} K"
    else:
        return f"₹{amount:.0f}"


def parse_indian_currency(text: str) -> float | None:
    """Parse text representation of Indian currency into numeric INR float.

    Supports:
        '45 Lakh', '1.5 Crore', '1.5 Cr', '75 L', '5000000', '₹ 1.2 Cr'
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.replace("₹", "").replace(",", "").strip().lower()
    match_cr = re.search(r"([\d\.]+)\s*(?:cr|crore|crores)", cleaned)
    if match_cr:
        return float(match_cr.group(1)) * 10_000_000

    match_lakh = re.search(r"([\d\.]+)\s*(?:lakh|lakhs|lac|lacs|l)", cleaned)
    if match_lakh:
        return float(match_lakh.group(1)) * 100_000

    match_k = re.search(r"([\d\.]+)\s*(?:k|thousand|thousands)", cleaned)
    if match_k:
        return float(match_k.group(1)) * 1_000

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_area_to_sqft(raw_val: str | float, default_unit: str = "sqft") -> float | None:
    """Normalize raw area inputs into square feet.

    Handles:
        - Numbers (float/int)
        - Ranges like '1200 - 1400' -> average 1300
        - Units: Sq. Meter, Sq. Yards (Gaj), Acres, Guntha, Ground, Cents
    """
    if raw_val is None:
        return None

    if isinstance(raw_val, (int, float)):
        return float(raw_val) if raw_val > 0 else None

    val_str = str(raw_val).strip().lower()
    if not val_str:
        return None

    # Handle range format like "1195 - 1440" or "1195-1440"
    if "-" in val_str:
        parts = val_str.split("-")
        try:
            low = float(re.findall(r"[\d\.]+", parts[0])[0])
            high = float(re.findall(r"[\d\.]+", parts[1])[0])
            return (low + high) / 2.0
        except (IndexError, ValueError):
            pass

    # Extract numerical part
    numbers = re.findall(r"[\d\.]+", val_str)
    if not numbers:
        return None

    try:
        val = float(numbers[0])
    except ValueError:
        return None

    # Unit conversions to square feet
    if "meter" in val_str or "sq. mt" in val_str or "sqm" in val_str:
        return val * 10.7639
    elif "yard" in val_str or "sq. yr" in val_str or "gaj" in val_str:
        return val * 9.0
    elif "acre" in val_str:
        return val * 43560.0
    elif "guntha" in val_str:
        return val * 1089.0
    elif "ground" in val_str:
        return val * 2400.0
    elif "cent" in val_str:
        return val * 435.6
    else:
        # Default unit check
        if default_unit.lower() in ("sq_yards", "sq yards", "gaj"):
            return val * 9.0
        elif default_unit.lower() in ("sq_meters", "sq meters", "sqm"):
            return val * 10.7639
        return val


def normalize_location_name(text: str | None) -> str:
    """Normalize a city or locality string (trim whitespace, remove excessive punctuation, title-case)."""
    if not text or not isinstance(text, str):
        return "Unknown"

    cleaned = re.sub(r"[^\w\s\-\.]", "", text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else "Unknown"
