# Dataset 06: Loan & Credit Risk

## 1. Problem & Objective
- **Problem**: Credit underwriting and retail loan default risk modeling.
- **ML Objective**:
  - **Binary Classification / Probability Estimation**: Predict `default` (0 = Fully Paid, 1 = Defaulted / Charged Off).

## 2. Row Granularity
Each row represents a single evaluated consumer loan application.

## 3. Data Dictionary

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `applicant_id` | String | Unique borrower application identifier | `APP_000001` |
| `income` | Integer | Annual verifiable income ($) | `$18,000` to `$450,000` |
| `loan_amount` | Integer | Requested loan principal ($) | `$2,000` to `$65,000` |
| `loan_term_months` | Integer | Repayment duration (36 or 60 months) | `36`, `60` |
| `interest_rate` | Float | Annual percentage rate (APR %) | `4.5%` to `32.0%` |
| `credit_score` | Integer | FICO credit rating score (380 to 850) | `380` to `850` |
| `employment_length_years` | Float | Years of verifiable work experience | `0.0` to `35.0`, `NaN` (7% missing) |
| `existing_debt` | Float | Outstanding total revolving + installment debt ($) | `$0` to `$380,000`, `NaN` (3% missing) |
| `monthly_payment` | Float | Calculated required monthly debt installment ($) | `$65.00` to `$2,150.00` |
| `dependents` | Integer | Number of financial dependents | `0`, `1`, `2`, `3`, `4` |
| `home_ownership` | Categorical | Housing status | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `loan_purpose` | Categorical | Stated use of funds | `debt_consolidation`, `credit_card`, `small_business`, etc. |
| `default` | Binary (0/1) | **Target**: 1 = Defaulted on loan, 0 = Paid in full | `0` (86.0%), `1` (14.0%) |

## 4. Intentionally Embedded Real-World Patterns
1. **Critical Financial Ratios**: Ratios such as Debt-to-Income (DTI) `(existing_debt/12 + monthly_payment) / (income/12)` and Payment-to-Income (PTI) are far stronger predictors than raw dollar amounts.
2. **Right-Skewed Wealth Distributions**: Highly skewed `income` and `existing_debt`.
3. **Non-Linear Risk Thresholds**: Default hazard escalates sharply when `credit_score < 600` or `DTI > 0.45`.
4. **Legitimate High-Debt Outliers**: High earners (e.g. $300k+ income) carrying $150k debt with prime 780+ credit scores have lower default rates than low earners with modest debt.
5. **Missingness in Income/Employment**: Gig economy / self-employed applicants missing formal employment tenure records.
