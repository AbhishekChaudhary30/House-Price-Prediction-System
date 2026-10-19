# Data Dictionary: Canonical Housing Schema

This document specifies the canonical schema, data types, physical measurement units, valid category values, and domain validation boundaries used across the **India House Price Prediction System**.

---

## 1. Schema Specifications

| Column Name | Data Type | Physical Unit | Mandatory | Domain Bounds / Accepted Categories | Description |
|---|---|---|---|---|---|
| `city` | `string` | N/A | Yes | `Bengaluru`, `Mumbai`, `Delhi` | Primary metropolitan territory of the residential property. |
| `locality` | `string` | N/A | Yes | Standardized Title Case (e.g., `Whitefield`, `Andheri West`) | Neighborhood or micro-market location. |
| `property_type` | `string` | N/A | Yes | `Apartment`, `Builder Floor`, `Independent House`, `Villa`, `Penthouse` | Architectural classification of the residential dwelling. |
| `bhk` | `int64` | Count | Yes | `[1, 10]` | Count of bedrooms (Bedroom-Hall-Kitchen configuration). |
| `bathrooms` | `float64` | Count | Yes | `[1.0, 10.0]` | Number of sanitary bathroom units. Defaults to `bhk` if missing. |
| `area_sqft` | `float64` | Square Feet | Yes | `[150.0, 25000.0]` | Standardized usable area in square feet. Converted from original units. |
| `floor_number` | `int64` | Floor index | Yes | `[0, 100]` | Floor on which the unit is located (0 = Ground floor). Must be $\le \text{total\_floors}$. |
| `total_floors` | `int64` | Count | Yes | `[1, 120]` | Total number of above-ground floors in the residential structure. |
| `furnishing_status` | `string` | N/A | Yes | `Unfurnished`, `Semi-Furnished`, `Furnished` | Interior furnishing level included with the property. |
| `parking_available` | `int64` | Binary (0/1) | Yes | `{0, 1}` | Indicator whether dedicated vehicular parking is provided. |
| `lift_available` | `int64` | Binary (0/1) | Yes | `{0, 1}` | Indicator whether passenger elevator access is present. |
| `gated_community` | `int64` | Binary (0/1) | Yes | `{0, 1}` | Indicator whether security gate / managed residential society applies. |
| `price_inr` | `float64` | Indian Rupees | Yes (Target) | `[300,000.0, 500,000,000.0]` | Canonical target: Advertised asking price in Indian Rupees (INR). |
| `price_type` | `string` | N/A | Metadata | `asking_price` | Label clarifying pricing nature (asking price vs deed price vs synthetic). |
| `source_id` | `string` | N/A | Metadata | e.g. `bengaluru_listings_kaggle` | Traceable data provenance key pointing to source documentation. |

---

## 2. Engineered Features (Generated Internally)

| Feature Name | Type | Formula | Interpretation |
|---|---|---|---|
| `area_per_bhk` | `float64` | `area_sqft / bhk` | Average space allocation per bedroom. Helps capture spacious vs compact layouts. |
| `floor_ratio` | `float64` | `floor_number / total_floors` | Relative vertical elevation in high-rise buildings (0.0 = ground, 1.0 = top floor). |
| `bath_to_bhk_ratio` | `float64` | `bathrooms / bhk` | Measure of luxury specification (e.g. 1 bathroom per BHK vs shared bathrooms). |

---

## 3. Data Integrity & Anti-Leakage Rules

1. **Target Leakage Prohibition**:
   - `price_per_sqft` derived from `price_inr / area_sqft` is **strictly forbidden** as an input feature during model training. It is only computed post-inference for descriptive reporting to the user.
   - Unique listing IDs, database row counters, and transaction completion statuses are excluded from feature vectors.
2. **Deterministic Preprocessing**:
   - Categorical imputation uses `'Unknown'` and numeric imputation uses training medians fitted inside the `ColumnTransformer`.
   - The test split and real-time user inputs never influence imputer or encoder parameters.
3. **Out-of-Distribution Rejection**:
   - If a requested property has an area $< 150$ sqft or $> 25,000$ sqft, or an invalid floor configuration ($\text{floor} > \text{total floors}$), validation fails proactively before reaching the model pipeline.
