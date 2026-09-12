# Dataset 03: Customer Transactions & Fraud Detection

## 1. Problem & Objective
- **Problem**: Real-time identification of fraudulent payment transactions.
- **ML Objective**:
  - **Imbalanced Binary Classification**: Predict `is_fraud` (0 = Legitimate, 1 = Fraudulent).

## 2. Row Granularity
Each row represents a single point-of-sale or digital e-commerce transaction.

## 3. Data Dictionary

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `transaction_id` | String | Unique transaction reference | `TXN_0000001` |
| `customer_id` | String | Account holder identifier (12,000 customers) | `CUST_000042` |
| `transaction_amount` | Float | Value of transaction ($) | `$1.00` to `$18,500.00` |
| `transaction_time` | String (Timestamp) | Timestamp of payment attempt | `2024-03-15 02:43:12` |
| `customer_age` | Float | Age of cardholder | `18` to `85`, `NaN` (4% missing) |
| `account_age_days` | Integer | Account tenure in days | `30` to `3,650` |
| `transaction_count_24h` | Integer | Number of attempts in previous 24 hours | `1` to `25` |
| `average_transaction_amount` | Float | Historical mean transaction size for customer ($) | `$25.00` to `$2,200.00` |
| `merchant_category` | Categorical | Retail sector | `Grocery`, `Luxury Goods`, `Cryptocurrency`, etc. |
| `device_type` | Categorical | Hardware / OS signature | `Mobile iOS`, `Unknown Bot`, `NaN` (3% missing) |
| `location` | Categorical | Geographical context | `Domestic Urban`, `International Online`, etc. |
| `previous_fraud_count` | Integer | Past recorded fraud incidents on account | `0`, `1` |
| `failed_attempts` | Integer | Failed PIN / OTP attempts before transaction | `0` to `5` |
| `is_fraud` | Binary (0/1) | **Target**: 1 = Confirmed Fraud, 0 = Legitimate | `0` (98.2%), `1` (1.8%) |

## 4. Intentionally Embedded Real-World Patterns
1. **Extreme Class Imbalance**: ~1.8% fraud prevalence.
2. **Behavioral Velocity & Time Patterns**: Fraud clusters heavily in late-night hours (01:00 - 05:00) with rapid succession bursts (`transaction_count_24h` > 8).
3. **Spend Deviation**: Strong signal in the ratio `transaction_amount / average_transaction_amount` (often > 5x for fraud).
4. **Legitimate High-Value Whales**: High net worth users with legitimate $5,000 transactions during daytime with zero failed attempts to prevent false positive traps.
5. **Card Testing Attacks**: Fraudsters testing stolen card validity with low $1.00 - $3.50 amounts.
