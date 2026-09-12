# EDA & Feature Engineering Technique Cheatsheet

A concise reference table covering all major techniques, their exact mathematical/algorithmic intent, diagnostic indicators, and clean Python implementations.

---

## 1. Univariate & Distribution Techniques

| Technique | WHAT | WHY | WHEN | WHEN NOT | Diagnostic Trigger | Python Implementation |
|---|---|---|---|---|---|---|
| **Histogram + KDE** | Density distribution plot | Inspect modality, symmetry, and tails | Always for continuous features | High-cardinality discrete IDs | Visualizing raw feature shape | `sns.histplot(df['col'], kde=True)` |
| **Boxplot & IQR** | Five-number summary + outlier bounds | Identify extreme spread and median | Continuous features with potential outliers | Highly discrete or multimodal distributions | Skewness $> 1.0$, long tails | `sns.boxplot(x=df['col'])` |
| **Log1p Transform** | $\ln(x + 1)$ mapping | Compress long right tails, stabilize variance | Right-skewed non-negative variables | Negative values or left-skewed distributions | `df['col'].skew() > 1.5` | `np.log1p(df['col'])` |
| **Yeo-Johnson** | Power transform for real numbers | Transform data with zeros/negatives toward normality | Skewed data containing negative numbers | When simple log is sufficient ($x > 0$) | Non-positive skewed continuous data | `PowerTransformer(method='yeo-johnson').fit_transform(X)` |
| **Missing Indicator** | Binary flag $I(x = \text{NaN})$ | Preserve the informational signal of missingness | When missingness is informative (MNAR / MAR) | When missingness is pure random noise (MCAR < 1%) | Feature missingness $> 5\%$ | `df['col_isna'] = df['col'].isna().astype(int)` |

---

## 2. Bivariate & Multivariate Relationship Techniques

| Technique | WHAT | WHY | WHEN | WHEN NOT | Diagnostic Trigger | Python Implementation |
|---|---|---|---|---|---|---|
| **Scatter + Trendline** | Point plot with regression fit | Identify linearity, curvature, or heteroscedasticity | Numerical vs Numerical (Regression) | Categorical variables with low distinct points | Investigating feature-target relationship | `sns.regplot(data=df, x='feat', y='target', scatter_kws={'alpha':0.2})` |
| **Grouped Boxplot / Violin** | Distribution per category | Compare variance and median shifts across levels | Categorical vs Numerical | Cardinality $> 20$ levels | Assessing category separation on target | `sns.boxplot(data=df, x='cat_col', y='num_col')` |
| **Contingency Table (Crosstab)** | Frequency count across categories | Detect association and class imbalance across categories | Categorical vs Categorical | Continuous numerical variables | Evaluating category-target interaction | `pd.crosstab(df['cat_1'], df['target'], normalize='index')` |
| **Correlation Heatmap** | Pearson / Spearman correlation matrix | Spot pairwise linear / monotonic collinearity | Continuous numerical feature sets | Non-linear relationships or unranked categoricals | Multicollinearity screening | `sns.heatmap(df.corr(method='spearman'), annot=True, cmap='vlag')` |
| **Variance Inflation Factor (VIF)** | Collinearity diagnostic ($1 / (1 - R_i^2)$) | Quantify how much variance of a coef is inflated by collinearity | Linear / Logistic Regression model prep | Tree models (trees handle collinearity naturally) | Pairwise correlation $|r| > 0.80$ | `variance_inflation_factor(X.values, i)` |

---

## 3. Categorical Encoding Techniques

| Technique | WHAT | WHY | WHEN | WHEN NOT | Diagnostic Trigger | Python Implementation |
|---|---|---|---|---|---|---|
| **One-Hot Encoding** | Binary indicator per level | Represent nominal categories without false ordering | Low cardinality ($\le 10$ levels) | High cardinality ($> 50$ levels) | Nominal categorical feature | `OneHotEncoder(drop='first', sparse_output=False)` |
| **Ordinal Encoding** | Integer mapping preserving rank | Preserve meaningful monotonic rank | Categories with intrinsic order (e.g. Low/Med/High) | Nominal categories with no order | Natural hierarchy in category | `OrdinalEncoder(categories=[['Low', 'Med', 'High']])` |
| **Frequency / Count Encoding** | Replace level with its sample share | Compact representation of popularity/rarity | Medium-to-high cardinality nominal features | Low cardinality where OHE is better | High cardinality with distribution signals | `df['cat_freq'] = df['cat'].map(df['cat'].value_counts(normalize=True))` |
| **Target Encoding (Smoothed)** | Replace category with target mean + shrinkage | Dense numerical representation directly aligned with target | High cardinality nominal features | Small datasets without smoothing (leaks target) | Cardinality $> 30$ in supervised learning | `TargetEncoder(smooth="auto", cv=5)` |

---

## 4. Time-Series & Temporal Techniques

| Technique | WHAT | WHY | WHEN | WHEN NOT | Diagnostic Trigger | Python Implementation |
|---|---|---|---|---|---|---|
| **Cyclical Encoding** | Sine & Cosine projection | Preserve circular continuity (e.g., 23:00 is close to 00:00) | Hour, Day of Week, Month in Linear/NN models | Tree-based models (trees split on raw integers easily) | Periodic temporal features | `np.sin(2*np.pi*df['hour']/24)`, `np.cos(...)` |
| **Lag Features** | Prior timestep value $y_{t-k}$ | Capture autoregressive momentum and dependency | Time-series forecasting and sensor monitoring | Cross-sectional independent rows | Autocorrelation function (ACF) peak | `df.groupby('entity')['val'].shift(1)` |
| **Rolling Window Statistics** | Moving average / std over window $W$ | Smooth out high-frequency noise and capture local trends | Sensor signals, financial time-series | Static non-temporal datasets | High local volatility / drift | `df.groupby('entity')['val'].transform(lambda x: x.shift(1).rolling(7).mean())` |

---

## 5. Scaling & Normalization Techniques

| Technique | WHAT | WHY | WHEN | WHEN NOT | Diagnostic Trigger | Python Implementation |
|---|---|---|---|---|---|---|
| **StandardScaler** | $(x - \mu) / \sigma$ | Center to zero mean and unit variance | Linear/Logistic regression, SVM, KNN, PCA | Data with extreme outliers; Tree models | Gradient/distance-based algorithms | `StandardScaler().fit_transform(X_train)` |
| **RobustScaler** | $(x - Q_2) / (Q_3 - Q_1)$ | Scale using median and IQR; unaffected by outliers | Continuous features with genuine extreme values | Strictly bounded inputs required | Presence of heavy tails/outliers in linear/KNN | `RobustScaler().fit_transform(X_train)` |
| **MinMaxScaler** | $(x - x_{min}) / (x_{max} - x_{min})$ | Rescale features strictly into $[0, 1]$ interval | Neural Networks, image inputs, bounded bounds | Extreme outliers present (compresses normal range) | Algorithm requires positive bounded inputs | `MinMaxScaler().fit_transform(X_train)` |
