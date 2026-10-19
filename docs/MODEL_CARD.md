# Model Card: India House Price Regression Model (v1.0.0)

## 1. Model Details
- **Model Identifier**: `Hist_Gradient_Boosting_v1.0`
- **Model Type**: Histogram-based Gradient Boosted Decision Tree Regressor (`HistGradientBoostingRegressor`) wrapped in `TransformedTargetRegressor` with $\log(1 + y)$ target transformation and $\exp(y) - 1$ inverse transformation.
- **Framework**: `scikit-learn` (v1.7.2)
- **Serialization**: `joblib` compressed artifact (`artifacts/house_price_pipeline_v1.joblib`)
- **Developer**: Abhishek Chaudhary (Senior ML Engineer Portfolio Project)
- **Release Date**: October 2026
- **Status**: Production-ready for Streamlit interactive inference.

---

## 2. Intended Use & Target Users
- **Primary Use Case**: Estimate fair asking price in Indian Rupees (INR) for residential properties (Apartments, Builder Floors, Independent Houses, Villas) in supported Indian metropolitan markets.
- **Intended Audience**: Homebuyers, prospective real-estate investors, financial analysts, and ML portfolio evaluators.
- **Prohibited / Out-of-Scope Uses**:
  - Commercial underwriting or automated mortgage sanction without physical survey.
  - Speculative high-frequency property trading.
  - Valuation in unsupported cities without local dataset integration.

---

## 3. Training Data & Preprocessing
- **Source Observations**: 55,543 cleaned records from verified listing sources across **Bengaluru**, **Mumbai**, and **Delhi**.
- **Data Split**:
  - **Train Set (80%)**: 44,434 properties.
  - **Held-Out Test Set (20%)**: 11,109 properties (stratified by city).
- **City Representation**:
  - Mumbai (MMR): 75.8% (42,105 properties)
  - Bengaluru: 22.2% (12,328 properties)
  - Delhi (NCR): 2.0% (1,110 properties)
- **Pipeline Architecture**:
  1. `HousingFeatureEngineer`: Computes `area_per_bhk`, `floor_ratio`, and `bath_to_bhk_ratio`.
  2. `ColumnTransformer`:
     - Numeric: `SimpleImputer(strategy='median')` $\rightarrow$ `RobustScaler()`
     - Categorical: `SimpleImputer(strategy='constant', fill_value='Unknown')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', min_frequency=5)`
     - Binary: `SimpleImputer(strategy='most_frequent')`
  3. `TransformedTargetRegressor`: Stabilizes target variance and prevents negative price outputs.

---

## 4. Evaluation Results (Held-Out Test Set: 11,109 Properties)

### Candidate Model Comparison

| Model | Test MAE | Test RMSE | Median AE | Test R² | MAPE (%) | Status |
|---|---|---|---|---|---|---|
| `Baseline_Median` | ₹96.77 Lakh | ₹2.29 Cr | ₹44.00 Lakh | -0.0759 | 78.57% | Naive Baseline |
| `Ridge_Regression` | ₹47.28 Lakh | ₹2.97 Cr | ₹14.02 Lakh | -0.8037 | 28.55% | Linear Benchmark |
| `Random_Forest` | ₹55.03 Lakh | ₹1.36 Cr | ₹22.72 Lakh | 0.6203 | 39.62% | Bagging Ensemble |
| **`Hist_Gradient_Boosting`** | **₹43.65 Lakh** | **₹1.19 Cr** | **₹17.41 Lakh** | **0.7108** | **29.80%** | **Champion Model** |

### Geographic Slice Analysis (Champion Model)

| City | Test Samples | Test MAE | Median Absolute Error | Test R² |
|---|---|---|---|---|
| **Bengaluru** | 2,465 | ₹35.42 Lakh | ₹14.48 Lakh | 0.5790 |
| **Delhi** | 222 | ₹66.45 Lakh | ₹33.87 Lakh | 0.7029 |
| **Mumbai** | 8,422 | ₹45.46 Lakh | ₹18.24 Lakh | 0.7239 |

### Property Type Slice Analysis

| Property Type | Test Samples | Test MAE | Median Absolute Error | Test R² |
|---|---|---|---|---|
| **Apartment** | 10,508 | ₹41.11 Lakh | ₹16.52 Lakh | **0.7251** |
| **Builder Floor** | 120 | ₹77.62 Lakh | ₹33.81 Lakh | 0.6835 |
| **Independent House** | 391 | ₹79.67 Lakh | ₹43.85 Lakh | 0.5043 |
| **Villa** | 90 | ₹1.38 Cr | ₹43.52 Lakh | 0.5791 |

---

## 5. Limitations & Ethical Caveats
1. **Asking Price Bias**: Advertised portal listing prices are subject to seller optimism and negotiation buffers (typically 5–15% above final sale deeds).
2. **Geographic Imbalance**: Mumbai comprises ~76% of the training dataset. While Bengaluru and Delhi have thousands of observations, hyper-local micro-markets with fewer than 3 samples receive broader city-level fallback pricing with an expanded $\pm 25\%$ prediction interval.
3. **Temporal Drift**: Property prices in Indian tier-1 cities experience market appreciation over time. Historical records from 2018–2021 serve as base pricing benchmarks and do not replace live on-ground broker appraisal.
