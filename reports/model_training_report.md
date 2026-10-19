# India House Price Prediction — Model Training & Evaluation Report

## 1. Candidate Model Performance Comparison (Held-Out Test Set)

| Model Name | Test MAE | Test RMSE | Median AE | Test R² | MAPE (%) |
|---|---|---|---|---|---|
| `Baseline_Median` | ₹96.77 Lakh | ₹2.29 Cr | ₹44.00 Lakh | -0.0759 | 78.57% |
| `Ridge_Regression` | ₹47.28 Lakh | ₹2.97 Cr | ₹14.02 Lakh | -0.8037 | 28.55% |
| `Hist_Gradient_Boosting` **(Champion)** | ₹43.65 Lakh | ₹1.19 Cr | ₹17.41 Lakh | 0.7108 | 29.80% |
| `Random_Forest` | ₹55.03 Lakh | ₹1.36 Cr | ₹22.72 Lakh | 0.6203 | 39.62% |

## 2. Slice Error Analysis (Champion Model)

### A. Performance by City

| City | Samples | MAE | Median AE | R² |
|---|---|---|---|---|
| **Bengaluru** | 2465 | ₹35.42 Lakh | ₹14.48 Lakh | 0.5790 |
| **Delhi** | 222 | ₹66.45 Lakh | ₹33.87 Lakh | 0.7029 |
| **Mumbai** | 8422 | ₹45.46 Lakh | ₹18.24 Lakh | 0.7239 |

### B. Performance by Price Segment

| Price Segment | Samples | MAE | Median AE | R² |
|---|---|---|---|---|
| **Budget (< 50L)** | 2818 | ₹11.48 Lakh | ₹8.05 Lakh | -1.7512 |
| **Luxury (> 4Cr)** | 686 | ₹2.84 Cr | ₹1.79 Cr | 0.2688 |
| **Mid-Market (50L - 1.5Cr)** | 5235 | ₹22.16 Lakh | ₹15.60 Lakh | -0.5796 |
| **Premium (1.5Cr - 4Cr)** | 2370 | ₹59.94 Lakh | ₹48.60 Lakh | -0.4701 |

### C. Performance by Property Type

| Property Type | Samples | MAE | Median AE | R² |
|---|---|---|---|---|
| **Apartment** | 10508 | ₹41.11 Lakh | ₹16.52 Lakh | 0.7251 |
| **Builder Floor** | 120 | ₹77.62 Lakh | ₹33.81 Lakh | 0.6835 |
| **Independent House** | 391 | ₹79.67 Lakh | ₹43.85 Lakh | 0.5043 |
| **Villa** | 90 | ₹1.38 Cr | ₹43.52 Lakh | 0.5791 |
