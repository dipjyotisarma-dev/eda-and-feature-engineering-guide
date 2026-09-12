# EDA Decision Tree: What to Investigate & Why

This guide provides a structured, objective-driven decision framework to determine exactly **what to check**, **why to check it**, and **what decisions follow** when encountering an unfamiliar dataset.

---

## 1. The Core EDA Decision Loop

```
┌────────────────────────────────────────────────────────┐
│ 1. DATASET & ML OBJECTIVE                              │
│    What is the target? (Regression / Classification /  │
│    Time-Series / Anomaly / Unsupervised)               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. FEATURE TAXONOMY IDENTIFICATION                     │
│    Numerical Continuous | Discrete Count | Categorical │
│    Nominal | Categorical Ordinal | Datetime | ID / Text│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. SYSTEMATIC EXPLORATION                              │
│    Univariate  ──► Bivariate  ──► Multivariate ──►     │
│    Target-Aware ──► Data Quality & Leakage Checks      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. DECISION & PIPELINE ACTION                          │
│    Because we observed [Finding],                      │
│    We will perform [Transformation/Encoding/Imputation]│
│    And avoid [Inappropriate Method]                    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Feature Type Decision Tree

### A. Numerical Continuous & Discrete Features

```
Numerical Feature
  │
  ├── [Check 1: Missingness & Sparsity]
  │     ├── Is missing % > 60% and uninformative? ──► ACTION: Consider dropping or test missing indicator.
  │     ├── Is missing % low (< 5%) & MAR/MCAR? ──► ACTION: Impute median (robust) or mean (normal).
  │     └── Is missingness tied to a state? ──► ACTION: Create explicit `is_missing` flag + domain imputation.
  │
  ├── [Check 2: Distribution & Skewness]
  │     ├── Absolute Skewness > 1.5 & positive values? ──► ACTION: Test `np.log1p(x)` or Box-Cox / Yeo-Johnson.
  │     ├── Multimodal distribution? ──► ACTION: Investigate sub-populations or tree-based binning.
  │     └── Heavy-tailed with extreme ranges? ──► ACTION: Quantile transformation or RobustScaler.
  │
  ├── [Check 3: Outliers & Extremes]
  │     ├── Domain error (e.g., negative mileage, age > 120)? ──► ACTION: Drop row or replace with NaN.
  │     ├── Legitimate rare event (e.g., luxury car, whale purchase)? ──► ACTION: Keep for trees, winsorize/cap for linear/distance models.
  │     └── High leverage point in regression? ──► ACTION: Evaluate Cook's distance and test log-scale target.
  │
  └── [Check 4: Zero Inflation & Constant Values]
        ├── High % of zeros (> 40%) (e.g., rainfall, discount)? ──► ACTION: Create binary indicator `has_feature` + continuous `log1p(feature)`.
        └── Zero variance (constant)? ──► ACTION: Drop feature immediately.
```

---

### B. Categorical Features (Nominal & Ordinal)

```
Categorical Feature
  │
  ├── [Check 1: Cardinality (Number of Unique Levels)]
  │     ├── Low Cardinality (2 to 10 unique levels)? ──► ACTION: One-Hot Encoding (OHE) with `drop_first=True` for linear models.
  │     ├── Moderate Cardinality (11 to 50 levels)? ──► ACTION: Group rare categories (< 1% frequency) into "Other" ──► OHE / Target Encoding.
  │     └── High Cardinality (> 50 levels, e.g. ZIP code, customer ID)? ──► ACTION: Target Encoding (with K-fold out-of-fold smoothing) or Frequency/Count Encoding.
  │
  ├── [Check 2: Inherent Natural Ordering]
  │     ├── True natural order (e.g. Low < Med < High, Education level)? ──► ACTION: Ordinal Encoding (integer mapping preserving monotonic order).
  │     └── Nominal (no order, e.g. Store ID, Department)? ──► ACTION: Never use arbitrary integers for linear/distance models. Use OHE or Target Encoding.
  │
  └── [Check 3: Target Separation by Category]
        ├── Certain categories have 0% or 100% target rate? ──► ACTION: Check for rare category overfitting; apply Laplace/m-estimate smoothing.
        └── High category overlap with target? ──► ACTION: Check if category adds predictive value via Chi-Square or Mutual Information.
```

---

### C. Datetime & Temporal Features

```
Datetime Column
  │
  ├── [Check 1: Range, Granularity & Missing Gaps]
  │     ├── Missing calendar dates or telemetry dropouts? ──► ACTION: Resample / Forward-fill if continuous physical process.
  │     └── Irregular timestamps? ──► ACTION: Extract elapsed time since previous event (`delta_t`).
  │
  ├── [Check 2: Seasonality & Periodicity]
  │     ├── Daily / Weekly cycles? ──► ACTION: Extract `dayofweek`, `is_weekend`, `hour`. Use Cyclical Sine/Cosine encoding for neural/linear models.
  │     └── Yearly / Monthly cycles? ──► ACTION: Extract `month`, `quarter`, `is_holiday_season`.
  │
  └── [Check 3: Temporal Ordering & Stationarity]
        ├── Trending mean/variance over time? ──► ACTION: Differencing, rolling statistics (`rolling_mean_7d`), and strict time-based train/test splitting (NEVER random split).
        └── Lags needed? ──► ACTION: Shift features strictly backwards (`lag_1`, `lag_7`) to avoid future lookahead leakage.
```

---

## 3. EDA Checklist by ML Objective

### Regression Objectives (Continuous Target $y$)
1. **Inspect Target Distribution**:
   - Is $y$ right-skewed? (e.g., Price, Salary, Claim Amount) $\rightarrow$ Evaluate $\log(y)$ transformation.
   - Are there negative or zero values blocking log transforms? $\rightarrow$ Use $\log(y + 1)$ or Yeo-Johnson.
2. **Inspect Feature vs. Target Linearity**:
   - Scatter plots & residual plots.
   - If relationship is exponential (e.g., Car Price vs. Age), log-transform target or use polynomial/spline terms.
3. **Check for Heteroscedasticity**:
   - If error variance expands with larger predictions $\rightarrow$ Log-transform target or use Weighted Least Squares / Robust Regression.

### Classification Objectives (Binary/Multiclass Target $y$)
1. **Inspect Class Balance**:
   - Severe imbalance (< 5% positive class)? $\rightarrow$ Avoid accuracy metric. Check PR-AUC / ROC-AUC, stratified K-fold, and class weighting.
2. **Inspect Class-Conditional Distributions**:
   - Overlay KDEs / boxplots of continuous features across target classes.
   - Look for feature separation and class-specific outliers.
3. **Inspect Categorical Target Rates**:
   - Bar plots of positive rate by category level with confidence intervals.
   - Identify high-risk vs. low-risk segmentations.

---

## 4. Master Decision Summary Table

| Finding | Risk / Issue | Recommended Action | Pitfall to Avoid |
|---|---|---|---|
| **Right-skewed numerical feature** | Distorts linear weights, high leverage | Apply `np.log1p` or Yeo-Johnson | Do not apply log to negative numbers |
| **High missingness (> 50%)** | Noise, sparse signal | Evaluate if missingness is informative $\rightarrow$ Add `is_missing` flag | Do not impute arbitrary constant without indicator |
| **High cardinality nominal feature** | Dimensionality explosion with OHE | Target encoding with smoothing or frequency encoding | Do not target-encode on the entire dataset (leaks target) |
| **Outlier in physical telemetry** | Sensor glitch / corrupted record | Impute via local rolling median or forward-fill | Do not blindly drop rows in time series |
| **High correlation ($r > 0.85$) between two features** | Multicollinearity in linear models, redundancy | Drop one feature or combine via ratio/PCA | Tree models tolerate collinearity, but linear models destabilize |
| **Temporal ordering in data** | Future data leakage | Chronological split + backward-only lags/rolling windows | Never use `train_test_split(shuffle=True)` |
