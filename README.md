# 🏠 India House Price Prediction System

An end-to-end, production-grade Machine Learning system designed to provide transparent, data-driven valuations for residential properties across major Indian metropolitan markets (**Bengaluru**, **Mumbai**, **Delhi**).

Built using genuine listing data from over **55,500 verified property records**, this system avoids fabricated datasets, formulas, and synthetic numbers to deliver defensible valuations, empirical prediction intervals, and transparent location support.

---

## 🌟 Key Capabilities

### 1. Interactive Valuation Engine (ML Inference)
- **Genuine Machine Learning Engine**: Trained on 55,500+ real-world residential listings using `HistGradientBoostingRegressor` wrapped in log-transformed target regression ($R^2 = 0.7108$ on held-out test data).
- **Multi-City & Locality Registry**: Covers over 3,600 neighborhoods across Bengaluru, Mumbai (MMR), and Delhi (NCR).
- **Zero Target Leakage**: Feature engineering strictly excludes target-derived variables (e.g. price per sq.ft.) from model inputs.
- **Defensible Uncertainty Quantification**: Generates empirical prediction intervals based on test-set error residuals rather than arbitrary confidence percentages.
- **Strict Location Verification**: Rejects unsupported cities (e.g., Patna, Lucknow) explicitly, refusing to silently substitute proxy locations.
- **Scenario Simulator**: Real-time "What-If" sensitivity simulator to test how adding balconies, bathrooms, or changing floor levels impacts property valuation.

### 2. Dataset Intelligence & Interactive Workflow
- **Ultra-Premium Light Interface**: Executive light theme (`#F8FAFC`, `#FFFFFF`, deep navy typography, restrained indigo/blue accents, muted teal highlights) crafted for optimal readability and zero visual clutter.
- **7-Step Guided Workflow**: Intuitive linear workflow with interactive top step tracker, state preservation across Streamlit reruns, and validated Previous / Next controls.
- **Automated Schema Role Detection**: Automatically maps incoming CSV columns to semantic roles (`price`, `city`, `locality`, `area`, `bhk`, `bathrooms`, `floor`, etc.) and profiles missingness and data types.
- **Transparent Automated Data Cleaning**: Strips whitespace, standardizes casing, handles empty rows, resolves duplicate listings, filters physical impossibilities, and outputs an auditable before/after report with CSV download.
- **One-Click Automated Intelligence**: Instant data-driven narrative insights, cross-city distribution summaries, and benchmark metrics derived directly from real data.
- **Interactive Plotly Visualization Studio**: Dynamic interactive charts (Price Distribution Histogram, Cross-City Boxplot, Area vs. Price Scatter with hover tooltips, BHK Median Bar Chart, Feature Correlation Heatmap) paired with instant tabular filtering and CSV export.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ / 3.13 |
| **Data Processing** | Pandas, NumPy, Scipy |
| **Machine Learning** | Scikit-learn (HistGradientBoosting, RandomForest, Ridge), Joblib |
| **Interactive Visualizations** | Plotly (px, go, ff), Matplotlib, Seaborn |
| **User Interface** | Streamlit (Custom CSS design system, session-state workflow router) |
| **Automated Testing** | Pytest (29 automated unit, integration, & Streamlit AppTests) |
| **Code Quality** | Ruff (0 errors, strict formatting & linting) |

---

## 📁 Project Architecture

```
house_price_prediction_system/
├── app.py                     # Streamlit application router & workflow orchestrator
├── requirements.txt           # Pinned production dependencies
├── .gitignore                 # Version control hygiene
├── LICENSE                    # MIT License
│
├── src/                       # Core machine learning & UI package
│   ├── config/settings.py     # Configuration, canonical schema & bounds
│   ├── data/
│   │   ├── data_sources.py    # Data source catalog & ingestion
│   │   ├── load_data.py       # Raw & processed data loaders
│   │   ├── clean_data.py      # Cleaning, unit normalization & filtering
│   │   ├── validate_data.py   # Schema validation & quality report
│   │   ├── prepare_features.py# Pipeline orchestrator & location registry
│   │   └── eda.py             # Reproducible EDA chart generation
│   ├── features/
│   │   └── feature_pipeline.py# Leakage-safe transformers & encoders
│   ├── models/
│   │   ├── train.py           # Multi-model training & slice evaluation
│   │   ├── evaluate.py        # Regression metrics & slice analysis
│   │   ├── predict.py         # Raw model inference helper
│   │   └── model_registry.py  # Model serialization & deserialization
│   ├── services/
│   │   └── prediction_service.py # Input validation, location checks, pricing service
│   ├── ui/                    # Premium Light UI & Analytics Modules (Part 2)
│   │   ├── styles.py          # Custom executive CSS design system & cards
│   │   ├── schema_detector.py # Automatic column role inference & profiling
│   │   ├── auto_cleaner.py    # Automated data cleaner & before/after audit
│   │   ├── analytics_engine.py# One-click narrative analytics & statistics
│   │   ├── chart_studio.py    # Plotly interactive visualization builders
│   │   └── views/             # 7-Step workflow views
│   │       ├── home_view.py       # Step 1: Landing overview & live status
│   │       ├── valuation_view.py  # Step 2: Interactive valuation & scenario simulator
│   │       ├── workspace_view.py  # Step 3: Dataset upload & schema detection
│   │       ├── cleaning_view.py   # Step 4: Automated data cleaning & audit
│   │       ├── insights_view.py   # Step 5: One-click automated analytics
│   │       ├── explorer_view.py   # Step 6: Visualization studio & filter table
│   │       └── about_view.py      # Step 7: Model card & methodology disclosures
│   └── utils/
│       ├── formatters.py      # Indian currency (Lakhs/Crores) & unit parsers
│       └── logging_config.py  # Structured logging configuration
│
├── data/
│   ├── raw/                   # Raw downloaded CSV files (git-ignored)
│   └── processed/             # Cleaned multi-city dataset (india_housing_processed.csv)
│
├── artifacts/                 # Versioned model pipelines & location registry
│   ├── house_price_pipeline_v1.joblib
│   ├── model_metadata_v1.json
│   └── location_registry_v1.json
│
├── reports/                   # Model training report & EDA charts
│   ├── model_training_report.md
│   ├── eda_price_distribution.png
│   ├── eda_price_by_city.png
│   ├── eda_area_vs_price.png
│   └── eda_correlation_matrix.png
│
├── docs/                      # Formal engineering documentation
│   ├── DATA_SOURCES.md        # Provenance, licensing & limitations
│   ├── DATA_DICTIONARY.md     # Canonical schema specifications
│   ├── MODEL_CARD.md          # Model card with measured evaluation metrics
│   └── PROJECT_STATUS.md      # Development milestones & roadmap
│
└── tests/                     # Automated test suite (29 unit, integration & UI tests)
    ├── conftest.py
    ├── test_data_validation.py
    ├── test_feature_pipeline.py
    ├── test_model_prediction.py
    ├── test_prediction_service.py
    ├── test_part2_features.py # Part 2 UI/UX, cleaner & analytics tests
    └── test_ui_apptest.py     # Streamlit AppTest automated UI & navigation suite
```

---

## 🧭 7-Step Guided Workflow Architecture

The application is structured into a stateful, linear 7-step guided pipeline with mutual state preservation:

1. **Step 1: Home & Overview** — System capability briefing, live model and dataset status indicators, quick-start action buttons.
2. **Step 2: Valuation Engine** — Interactive property valuation form with micro-market locality registry, empirical prediction intervals, real-time What-If sensitivity simulator, and valuation card export (JSON).
3. **Step 3: Dataset Workspace** — Load bundled verified 55.5k-listing dataset or upload external real estate CSVs with automated semantic schema role detection and missingness profiling.
4. **Step 4: Data Quality & Cleaning** — Transparent automated cleaning (whitespace standardization, duplicate removal, physical bounds validation) with before/after metric cards and cleaned CSV download.
5. **Step 5: Automated Intelligence** — Single-click execution generating instant narrative insights, city-level comparisons, and statistical distribution summaries.
6. **Step 6: Explore & Visualize** — Plotly Interactive Studio (Histograms, Boxplots, Scatter with tooltips, BHK Bars, Heatmaps) alongside manual text query search, numeric sliders, and filtered data export.
7. **Step 7: Model Card & Methodology** — Production model specs, slice-level metrics, empirical interval derivation, and ethical disclosures.

---

## ⚡ Quick Start & Local Execution

### 1. Installation
Clone the repository and install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Ingest Data & Generate Processed Dataset
To download the verified real-estate listing feeds, clean the records, and build the location registry:
```bash
python -m src.data.prepare_features
```

### 3. Generate Exploratory Data Analysis (EDA) Charts
Generate statistical distribution and correlation figures in `reports/`:
```bash
python -m src.data.eda
```

### 4. Train and Evaluate Candidate Models
Train baseline (`DummyRegressor`), linear benchmark (`Ridge`), `RandomForest`, and champion `HistGradientBoosting` models:
```bash
python -m src.models.train
```

### 5. Run Automated Tests
Execute the automated test suite across validation, feature engineering, persistence, and service layers:
```bash
python -m pytest tests/ -v
```

### 6. Launch the Streamlit Web Application
Run the interactive prediction dashboard locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 7. Deploy to Streamlit Community Cloud
This repository is pre-configured for one-click deployment:
1. Push this repository to your GitHub account (`git push origin main`).
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **Create app** and select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Advanced settings...** and ensure the **Python version** is set to **3.11** (matching the `.python-version` file and modern `scikit-learn 1.7` / `numpy` binary wheels).
5. Click **Deploy!** (If previously deployed, select **Manage app -> Reboot -> Clear cache and redeploy**).
6. Total repository and artifact weight is ~15 MB, well below the 1 GB cloud memory budget.

---

## 📊 Evaluation Results (Held-Out Test Set)

The models were evaluated on **11,109 held-out property listings** stratified across cities:

| Model | Test MAE | Test RMSE | Median Absolute Error | Test $R^2$ | MAPE |
|---|---|---|---|---|---|
| `Baseline_Median` | ₹96.77 Lakh | ₹2.29 Cr | ₹44.00 Lakh | -0.0759 | 78.57% |
| `Ridge_Regression` | ₹47.28 Lakh | ₹2.97 Cr | ₹14.02 Lakh | -0.8037 | 28.55% |
| `Random_Forest` | ₹55.03 Lakh | ₹1.36 Cr | ₹22.72 Lakh | 0.6203 | 39.62% |
| **`Hist_Gradient_Boosting` (Champion)** | **₹43.65 Lakh** | **₹1.19 Cr** | **₹17.41 Lakh** | **0.7108** | **29.80%** |

---

## ⚠️ Known Limitations & Scope

1. **Asking Price Bias**: Current verified feeds reflect advertised asking prices from property portals. Final transaction deed prices typically include a 5–15% negotiation discount.
2. **Coverage Scope (v1.0)**: Fully supported cities are **Bengaluru**, **Mumbai (MMR)**, and **Delhi (NCR)**. Other cities will be integrated as verified, clean datasets become available.
3. **No Live Scraping / No Paid APIs**: Built entirely on free and open-source tooling without requiring paid subscriptions or fragile scrapers.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
