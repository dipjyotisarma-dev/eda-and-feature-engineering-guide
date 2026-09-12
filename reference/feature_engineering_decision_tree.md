# Feature Engineering Decision Tree: What to Create & Why

This reference guide provides a definitive decision tree for selecting, engineering, and validating features based on **feature type**, **domain meaning**, and **downstream machine learning model**.

---

## 1. Feature Engineering by ML Model Class

Before engineering features, always evaluate the model family:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ML MODEL FAMILY REQUIREMENTS                                            │
├────────────────────────────────┬────────────────────────────────────────┤
│ Linear / Logistic Models       │ • Scaling is MANDATORY                 │
│                                │ • Skewness reduction (log1p/Yeo-Johnson)│
│                                │ • Must explicitly create interactions  │
│                                │ • Must handle collinearity / high VIF  │
├────────────────────────────────┼────────────────────────────────────────┤
│ Distance Models (KNN, SVM, PCA)│ • Scaling is CRITICAL (Standard/Robust)│
│                                │ • Highly sensitive to irrelevant noise │
│                                │ • Curse of dimensionality with high OHE│
├────────────────────────────────┼────────────────────────────────────────┤
│ Tree-Based (XGBoost, LightGBM, │ • Invariant to monotonic scaling       │
│ Random Forest, CatBoost)       │ • Naturally captures non-linear splits │
│                                │ • Benefits from ratios, domain aggregations│
│                                │ • Avoid excessive one-hot encoding     │
├────────────────────────────────┼────────────────────────────────────────┤
│ Neural Networks                │ • Scaling is MANDATORY (MinMax / Std)  │
│                                │ • Cyclical encoding for datetime/angles│
│                                │ • Dense embeddings for high cardinality│
└────────────────────────────────┴────────────────────────────────────────┘
```

---

## 2. Decision Trees by Transformation & Engineering Type

### A. Transformations (Log, Power, Scaling)

```
Numerical Variable $x$
  │
  ├── Is $x$ heavily right-skewed ($skew > 1.0$) with strictly non-negative values?
  │     ├── YES (includes zeros) ──► `np.log1p(x)` or `scipy.stats.yeojohnson`
  │     ├── YES (strictly positive $x > 0$) ──► `np.log(x)` or `scipy.stats.boxcox`
  │     └── NO (roughly symmetric or multi-modal) ──► Leave raw or standard scale
  │
  └── Which Scaling Method Should I Use?
        ├── Presence of extreme, legitimate outliers? ──► `RobustScaler` (uses median & IQR)
        ├── Strictly bounded features required (e.g. Image pixels, Neural Net input)? ──► `MinMaxScaler`
        ├── Normal-like features for Linear/Logistic/SVM/KNN? ──► `StandardScaler` (zero mean, unit variance)
        └── Tree-based ensemble (XGBoost / LightGBM)? ──► Scaling NOT required.
```

---

### B. Categorical Encoding

```
Categorical Variable $c$
  │
  ├── Does $c$ have an intrinsic natural order (e.g. Low, Medium, High)?
  │     ├── YES ──► `OrdinalEncoder` with explicit ordered mapping dictionary.
  │     └── NO (Nominal) ──► Evaluate Cardinality:
  │           │
  │           ├── Cardinality is LOW (<= 10 levels)
  │           │     ├── Linear / Logistic Model ──► `OneHotEncoder(drop='first', sparse_output=False)`
  │           │     └── Tree Model / Distance Model ──► `OneHotEncoder(drop=None)`
  │           │
  │           ├── Cardinality is MEDIUM (11 to 50 levels)
  │           │     ├── Group tail categories (< 1% freq) into `"Other"` ──► One-Hot Encode
  │           │     └── Or use `FrequencyEncoder` (replace level with % share)
  │           │
  │           └── Cardinality is HIGH (> 50 levels, e.g. ZIP code, Merchant ID)
  │                 ├── Target Encoding with K-Fold out-of-fold smoothing + m-estimate shrinkage
  │                 ├── Frequency / Count Encoding
  │                 └── CatBoost / LightGBM native categorical handling
```

---

### C. Ratio, Difference & Domain-Specific Features

```
Pairs of Numerical Variables $(A, B)$
  │
  ├── Does the ratio or difference capture a known business / physical principle?
  │     ├── Financial Burden: `debt / income` (DTI), `monthly_payment / income` (PTI)
  │     ├── Inventory Coverage: `inventory / average_daily_sales` (Days of Supply)
  │     ├── Velocity / Deviation: `transaction_amount / customer_avg_spend`
  │     ├── Tenure / Career Progress: `years_at_company - years_in_role`
  │     └── Efficiency / Utilization: `units_produced / operating_hours`, `load / max_capacity`
  │
  └── Safety Rules:
        ├── Division by Zero? ──► Always use `A / (B + epsilon)` or `np.where(B == 0, 0, A / B)`
        └── Check correlation with parent features ──► Ensure ratio provides new orthogonal signal.
```

---

### D. Datetime & Time-Series Feature Engineering

```
Datetime Timestamp $t$
  │
  ├── Calendar & Cyclic Components:
  │     ├── Day of Week, Month, Quarter, Hour of Day
  │     ├── For Linear/NN models: Convert cyclic components to Sine/Cosine:
  │     │     $x_{sin} = \sin(2\pi \cdot \text{hour} / 24)$, $x_{cos} = \cos(2\pi \cdot \text{hour} / 24)$
  │     └── Binary calendar flags: `is_weekend`, `is_month_end`, `is_holiday`
  │
  └── Sequential & Rolling Aggregations (Strictly Historical):
        ├── Lags: $y_{t-1}, y_{t-7}, y_{t-30}$ (captures autocorrelation and seasonality)
        ├── Rolling Windows: `rolling_mean_7d`, `rolling_std_7d`, `rolling_max_30d` (captures recent momentum/volatility)
        ├── Expanding Windows: `expanding_mean` (historical customer lifetime average)
        └── CRITICAL: Always shift by at least 1 step (`.shift(1)`) to avoid target leakage!
```

---

## 3. Decision Matrix: When to Engineer vs. When to Avoid

| Technique | When to Use | When NOT to Use / Risk |
|---|---|---|
| **Log Transformation** | Multiplicative variance, heavy right skew, strictly positive numbers. | Negative numbers, already symmetric features, tree models (no effect on split order). |
| **StandardScaler** | Distance/Gradient algorithms (Linear/Logistic, SVM, KNN, Neural Nets). | Tree models, data with heavy outliers (distorts mean/std $\rightarrow$ use `RobustScaler`). |
| **One-Hot Encoding** | Low-cardinality nominal variables ($\le 10$ levels). | High cardinality (> 50 levels) $\rightarrow$ creates massive sparse matrix and splits trees poorly. |
| **Target Encoding** | High cardinality categoricals where category shares predictive relationship with target. | Unsmoothed on small samples (overfits instantly); never compute across the full dataset before split! |
| **Domain Ratios** | When relative scale has business meaning (e.g. debt-to-income, days of inventory). | When denominator is frequently 0 or noisy; when linear models already have both raw variables and regularizer handles it. |
| **Binning / Discretization** | Non-linear step-function relationships, business interpretability (e.g., credit score tiers). | Arbitrary continuous data $\rightarrow$ loses variance and destroys information gradient. |
| **Polynomial Features** | Quadratic/interaction curves in linear/logistic regression. | High-dimensional data $\rightarrow$ combinatorial explosion ($O(p^2)$) and severe overfitting. |
| **Lag / Rolling Features** | Time-series forecasting and telemetry anomaly detection. | Cross-sectional independent datasets; future-lookahead windows (causes catastrophic leakage). |
