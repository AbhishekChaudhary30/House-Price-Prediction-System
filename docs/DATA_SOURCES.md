# Data Sources Documentation

This document records the provenance, licensing, collection methodology, schema mapping, and limitations of all real-estate datasets utilized by the **India House Price Prediction System**.

---

## 1. Ethical, Traceability, and Truthfulness Principles

1. **Zero Data Fabrication**: No synthetic transaction prices or fictitious housing sales records are passed off as real-world market transactions.
2. **Asking Prices vs. Completed Transactions**: All current primary data sources reflect advertised property listing prices (**asking prices**) collected from public property portals. Asking prices generally trade at a **5% to 15% premium** above final registered deed prices due to buyer negotiation and stamp duty disclosures.
3. **Transparent Fallbacks**: The system strictly restricts predictions to cities with verified training observations (**Bengaluru**, **Mumbai**, **Delhi**). Out-of-distribution cities (e.g., Patna, Jaipur, Lucknow) are explicitly identified as unsupported rather than silently estimated using an arbitrary proxy city.

---

## 2. Ingested Data Sources Catalog

### A. Bengaluru Property Listings Dataset
- **Source Identifier**: `bengaluru_listings_kaggle`
- **Collector / Author**: Amitabha Joy (Kaggle) / 99acres property listing portal aggregation.
- **Reference / URL**: [Bengaluru House Price Data (GitHub Raw)](https://raw.githubusercontent.com/tejas05in/pwskills-Data_Science_Masters/master/machine%20learning/assignments/Bengaluru_House_Data.csv)
- **License**: CC BY-SA 4.0 (Open Data for Research & Educational Use)
- **Observation Period**: 2016 – 2018 listings
- **Raw Observations**: 13,320 records
- **Retained Usable Records**: 12,328 records (after area parsing and outlier filtering)
- **Coverage**: Over 1,200 distinct localities across Bengaluru Urban & Suburban (e.g., Whitefield, Electronic City, Sarjapur Road, Hebbal, Thanisandra, Marathahalli, Yelahanka).
- **Price Target**: Advertised asking price listed in Lakhs INR (converted to canonical `price_inr` = `price * 100,000`).
- **Known Limitations**:
  - Raw feed does not record specific floor numbers or total building floors. These are imputed to standard multi-story apartment medians (floor 2 of 4) during ingestion.
  - Areas occasionally contain range formats (`1200 - 1400`) or non-sqft units (`Sq. Meter`, `Acres`), which are parsed and normalized to square feet via `parse_area_to_sqft`.

---

### B. Mumbai Metropolitan Region (MMR) Real Estate Appraisal Dataset
- **Source Identifier**: `mumbai_real_estate_appraisal`
- **Collector / Author**: Ashish Kumar / Mumbai Housing Portal Listings Repository.
- **Reference / URL**: [Mumbai Real Estate Appraisal Dataset (GitHub Raw)](https://raw.githubusercontent.com/ak1997/Comparison-of-Ensemble-Models-Using-Real-Estate-Appraisal/master/Mumbai.csv)
- **License**: Open Access for Academic & Portfolio Research
- **Observation Period**: 2018 – 2021 listings
- **Raw Observations**: 45,296 records
- **Retained Usable Records**: 42,105 records
- **Coverage**: Mumbai City, Mumbai Suburban, Thane, and Navi Mumbai (e.g., Andheri West, Bandra, Borivali, Chembur, Kharghar, Mira Road, Powai, Thane West).
- **Price Target**: Advertised asking price in Lakhs INR (`price * 100,000`).
- **Features Captured**: Carpet Area, Built-Up Area, Floor Number, Parking spaces, Furnishing, Bathrooms, Bedrooms (BHK).
- **Known Limitations**:
  - High variance in luxury segments (Cuffe Parade, Malabar Hill, Bandra West) where prices reach ₹40+ Crores.
  - Floor number is capped at 60 for statistical stability.

---

### C. Delhi NCR MagicBricks Housing Dataset
- **Source Identifier**: `delhi_magicbricks_listings`
- **Collector / Author**: Rohit Gupta (MagicBricks Portal Listings)
- **Reference / URL**: [Delhi Flat Price Prediction Repository (GitHub Raw)](https://raw.githubusercontent.com/rohit077/Delhi-Flat-Price-Prediction/master/MagicBricks.csv)
- **License**: MIT License (Educational Open Access)
- **Observation Period**: 2019 – 2020 listings
- **Raw Observations**: 1,259 records
- **Retained Usable Records**: 1,110 records
- **Coverage**: Delhi NCR neighborhoods (e.g., Rohini, Dwarka, Lajpat Nagar, Patel Nagar, Paschim Vihar, Greater Kailash).
- **Price Target**: Advertised listing price in raw INR.
- **Features Captured**: Area in sqft, BHK, Bathrooms, Furnishing, Parking count, Property Type (Apartment vs. Builder Floor).
- **Known Limitations & Anti-Leakage Measures**:
  - Raw dataset contains a `Per_Sqft` column calculated directly from `Price / Area`. To prevent **target leakage**, this column is explicitly stripped prior to any feature engineering or modeling.
  - Sample size is significantly smaller than Bengaluru and Mumbai feeds, resulting in wider prediction intervals for certain Delhi sub-localities.

---

## 3. Disqualified / Educational-Only Datasets

- **Kaggle Indian Urban House Price Prediction Dataset**:
  - Reference: `https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data`
  - Classification: **Synthetic** (Generated via synthetic mathematical formulas).
  - Status: **Disqualified** from primary model training. Not used as real market ground truth.

---

## 4. Ingestion & Preprocessing Summary

| Metric | Bengaluru | Mumbai | Delhi | Unified Multi-City |
|---|---|---|---|---|
| **Raw Listing Count** | 13,320 | 45,296 | 1,259 | 59,875 |
| **Cleaned & Validated Count** | 12,328 | 42,105 | 1,110 | **55,543** |
| **Removal Rate** | 7.4% | 7.0% | 11.8% | **7.2%** |
| **Localities Represented** | 1,298 | 2,130 | 184 | **3,612** |
| **Price Range Retained** | ₹12 L – ₹30 Cr | ₹8 L – ₹45 Cr | ₹14 L – ₹18 Cr | **₹8 L – ₹45 Cr** |
