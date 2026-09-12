# Exploratory Data Analysis (EDA) & Feature Engineering Decision Guide

A hands-on, decision-driven repository designed to teach you **how to decide WHAT EDA and FEATURE ENGINEERING to perform** based on dataset characteristics, feature taxonomy, target distribution, and downstream ML models.

---

## 🎯 The Core Philosophy: Decision-First ML

Most tutorials teach EDA as a mechanical checklist of plotting functions. This guide teaches the **analytical decision framework**:

```
DATASET & BUSINESS PROBLEM
         │
         ▼
FEATURE TAXONOMY (Continuous, Discrete, Nominal, Ordinal, Datetime)
         │
         ▼
WHAT SHOULD I CHECK? (Specific Hypotheses & Diagnostic Tests)
         │
         ▼
WHY SHOULD I CHECK IT? (Statistical & Domain Rationale)
         │
         ▼
WHAT DID I FIND? (Concrete Quantitative & Visual Observation)
         │
         ▼
WHAT ACTION SHOULD I TAKE? (Transformation, Imputation, Encoding, Pruning)
         │
         ▼
HOW DOES IT AFFECT THE ML PIPELINE? (Linear vs Tree vs Distance Sensitivity)
```

Every technique in this guide explicitly answers:
- **WHAT** is the technique?
- **WHY** are we using it?
- **WHEN** should it be used?
- **WHEN NOT** to use it?
- **WHAT TO LOOK FOR** in the data?
- **WHAT ACTION** does the finding lead to?

---

## 📊 Realistic Synthetic Datasets

| Dataset | Problem / ML Task | Target | Rows x Cols | Key Embedded Patterns |
|---|---|---|---|---|
| [**Retail Sales & Inventory**](datasets/retail/README.md) | Stockout Risk & Demand Forecasting | `stockout_risk` / `units_sold` | 60,000 x 15 | Weekly seasonality, promo lifts, lead-time deficits, stockout truncation. |
| [**Used Car Market**](datasets/used_cars/README.md) | Fair-Value Valuation (Regression) | `selling_price` | 22,000 x 14 | Exponential age depreciation, right-skewed price, rare models, luxury interactions. |
| [**Customer Fraud**](datasets/fraud/README.md) | Fraud Detection (Imbalanced Binary Cls) | `is_fraud` (1.8% rate) | 85,000 x 14 | Heavy spend deviations, 24h burst velocity, night transactions, card probing. |
| [**Employee Workforce**](datasets/workforce/README.md) | Attrition Prediction (Binary Cls) | `attrition` (16.2% rate) | 16,000 x 14 | Overtime + low satisfaction synergy, career stagnation ratios, tenure collinearity. |
| [**Industrial Machine Sensors**](datasets/machine_sensors/README.md) | Predictive Maintenance (Anomaly/Cls) | `failure` | 108,000 x 11 | Machine baselines, thermal/vibration drift, pre-breakdown stress precursors. |
| [**Loan & Credit Risk**](datasets/credit_risk/README.md) | Credit Underwriting (Binary Cls) | `default` (14.0% rate) | 52,000 x 13 | Debt-to-Income (DTI), Payment-to-Income, non-linear risk tiers, MNAR missingness. |

---

## 🗺️ Learning Path & Notebook Directory

The repository is structured into 6 logical learning phases:

### Phase 1: Foundational Frameworks & Relationships
- [`01_eda_decision_framework.ipynb`](notebooks/01_eda_decision_framework.ipynb): The 5-step EDA audit, automated feature taxonomy, and hypothesis formulation.
- [`02_univariate_eda_and_distributions.ipynb`](notebooks/02_univariate_eda_and_distributions.ipynb): Histograms, KDEs, boxplots, Fisher-Pearson skewness, and zero-inflation.
- [`03_bivariate_eda_and_relationships.ipynb`](notebooks/03_bivariate_eda_and_relationships.ipynb): Choosing plots for Num-Num, Cat-Num, and Cat-Cat pairs; linearizing non-linear decay.
- [`04_multivariate_eda_and_correlations.ipynb`](notebooks/04_multivariate_eda_and_correlations.ipynb): Pearson vs Spearman, Variance Inflation Factor (VIF), and interaction detection.

### Phase 2: Data Quality & Target-Aware Diagnostics
- [`05_missing_values_eda_and_treatment.ipynb`](notebooks/05_missing_values_eda_and_treatment.ipynb): Diagnosing MCAR vs MAR vs MNAR; median vs indicator vs category `'Unknown'`.
- [`06_outlier_detection_and_treatment.ipynb`](notebooks/06_outlier_detection_and_treatment.ipynb): Data entry bugs vs legitimate tail events; IQR fences, winsorizing, and power transforms.
- [`07_categorical_feature_eda.ipynb`](notebooks/07_categorical_feature_eda.ipynb): Cardinality audit, rare category consolidation (< 1%), and target rate lift.
- [`08_time_series_eda.ipynb`](notebooks/08_time_series_eda.ipynb): Chronological continuity, Autocorrelation (ACF), day-of-week seasonality, and drift.
- [`09_target_aware_eda_for_regression.ipynb`](notebooks/09_target_aware_eda_for_regression.ipynb): Target normality, heteroscedasticity fan-spread, and log-target modeling.
- [`10_target_aware_eda_for_classification.ipynb`](notebooks/10_target_aware_eda_for_classification.ipynb): Class imbalance (1.8%), class-conditional KDE shifts, and PR-AUC optimization.

### Phase 3: Feature Selection & Engineering Mechanics
- [`11_feature_types_and_feature_selection.ipynb`](notebooks/11_feature_types_and_feature_selection.ipynb): VarianceThreshold, Collinearity filtering, and Mutual Information (MI).
- [`12_numerical_feature_engineering.ipynb`](notebooks/12_numerical_feature_engineering.ipynb): Age differences, usage intensity, financial ratios, and luxury interactions.
- [`13_categorical_feature_engineering.ipynb`](notebooks/13_categorical_feature_engineering.ipynb): One-Hot vs Ordinal vs Frequency vs Smoothed Out-Of-Fold Target Encoding.
- [`14_datetime_and_time_series_features.ipynb`](notebooks/14_datetime_and_time_series_features.ipynb): Cyclical sine/cosine, grouped lag-7, and shifted rolling window statistics.
- [`15_ratio_interaction_and_domain_features.ipynb`](notebooks/15_ratio_interaction_and_domain_features.ipynb): DTI, Days of Supply, Runway Deficit, and Burnout Synergy scores.

### Phase 4: Model Dependencies & Leakage Prevention
- [`16_transformations_log_power_and_scaling.ipynb`](notebooks/16_transformations_log_power_and_scaling.ipynb): Skewness vs Normalization; empirical sensitivity across Linear, KNN, SVM, and Trees.
- [`17_binning_and_discretization.ipynb`](notebooks/17_binning_and_discretization.ipynb): Uniform vs Quantile vs Decision Tree binning; information loss trade-offs.
- [`18_encoding_strategies.ipynb`](notebooks/18_encoding_strategies.ipynb): Benchmarking encoding methods on memory footprint, training latency, and test RMSE.
- [`19_feature_reduction_and_redundancy.ipynb`](notebooks/19_feature_reduction_and_redundancy.ipynb): Hierarchical clustering, iterative VIF elimination, and permutation importance.
- [`20_data_leakage_and_train_only_preprocessing.ipynb`](notebooks/20_data_leakage_and_train_only_preprocessing.ipynb): Reproducing 5 fatal leakage bugs and constructing leak-proof Scikit-Learn pipelines.

### Phase 5: End-to-End Case Studies
- [`21_end_to_end_retail_eda_and_fe.ipynb`](notebooks/21_end_to_end_retail_eda_and_fe.ipynb): Retail inventory & stockout prediction from scratch.
- [`22_end_to_end_used_car_eda_and_fe.ipynb`](notebooks/22_end_to_end_used_car_eda_and_fe.ipynb): Used car valuation with non-linear depreciation and high-cardinality target encoding.
- [`23_end_to_end_fraud_eda_and_fe.ipynb`](notebooks/23_end_to_end_fraud_eda_and_fe.ipynb): Rare fraud classification with velocity features and PR-AUC tuning.
- [`24_end_to_end_machine_sensor_eda_and_fe.ipynb`](notebooks/24_end_to_end_machine_sensor_eda_and_fe.ipynb): Industrial sensor predictive maintenance and degradation precursor detection.

### Phase 6: Master Practitioner Playbook
- [`25_eda_fe_decision_playbook.ipynb`](notebooks/25_eda_fe_decision_playbook.ipynb): Interactive decision advisor and answers to 14 core real-world practitioner questions.

---

## 📚 Reference Guides

- [`reference/eda_decision_tree.md`](reference/eda_decision_tree.md): Visual decision tree for what to check based on data types and objectives.
- [`reference/feature_engineering_decision_tree.md`](reference/feature_engineering_decision_tree.md): Decision tree for selecting transformations, encodings, and domain features.
- [`reference/technique_cheatsheet.md`](reference/technique_cheatsheet.md): Master lookup table with WHAT/WHY/WHEN/WHEN NOT and Python snippets.
- [`reference/common_leakage_patterns.md`](reference/common_leakage_patterns.md): Prevention guide for 5 fatal data leakage vectors.

---

## 🚀 Quickstart

### 1. Clone and Install Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/eda-and-feature-engineering-guide.git
cd eda-and-feature-engineering-guide

# Install required Python packages
pip install -r requirements.txt
```

### 2. Generate All Synthetic Datasets
All datasets are generated reproducibly with fixed random seeds and embedded real-world imperfections (skew, outliers, missingness, non-linearities, temporal drift, and class imbalance):
```bash
python data_generation/generate_all.py
```

---

## ⚖️ License
MIT License. Created for practitioners seeking mastery over exploratory data analysis and feature engineering decision-making.
