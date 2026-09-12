# Common Data Leakage Patterns & Prevention Guide

Data leakage occurs when information from outside the training dataset (or from the future target state) is inadvertently used to create features or preprocess data. Leakage produces unrealistically high cross-validation scores that collapse in production.

---

## The 5 Most Fatal Leakage Patterns

### 1. Global Preprocessing Before Train/Test Split (The Baseline Trap)
* **What happens**: Calling `scaler.fit_transform(X)` or `imputer.fit_transform(X)` on the entire dataset before splitting into train and test sets.
* **Why it leaks**: The test set's mean $\mu_{test}$ and standard deviation $\sigma_{test}$ influence the training scaling parameters.
* **The Fix**: Always split first, then `fit()` on training data only:
```python
# WRONG (LEAKAGE):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Leaks test statistics
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# CORRECT (SAFE):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Transform only using train params
```

---

### 2. Unsmoothed Global Target Encoding (The Memorization Trap)
* **What happens**: Replacing high-cardinality category levels with the overall target mean across the whole dataset.
* **Why it leaks**: In rare categories with few instances, the category average directly encodes the specific target value of the row.
* **The Fix**: Use Out-Of-Fold (OOF) K-Fold target encoding fitted strictly within training folds, with Laplace or empirical Bayes smoothing:
```python
from sklearn.preprocessing import TargetEncoder

# CORRECT: TargetEncoder handles out-of-fold cross-validation internally
encoder = TargetEncoder(cv=5, smooth="auto", random_state=42)
X_train['category_encoded'] = encoder.fit_transform(X_train[['category']], y_train)
X_test['category_encoded'] = encoder.transform(X_test[['category']])
```

---

### 3. Future Temporal Lookahead in Rolling / Lag Features
* **What happens**: Calculating rolling averages without shifting backwards by at least 1 step (e.g. `df['sales'].rolling(7).mean()`).
* **Why it leaks**: In standard rolling operations, row $t$ includes the value at row $t$. If $t$ is the target you are forecasting today, you have just fed the current day's actual sales into the model input!
* **The Fix**: Always apply `.shift(1)` before computing rolling window statistics:
```python
# WRONG (LOOKAHEAD LEAKAGE):
df['rolling_7d_sales'] = df.groupby('store_id')['units_sold'].transform(lambda x: x.rolling(7).mean())

# CORRECT (STRICTLY HISTORICAL):
df['rolling_7d_sales'] = df.groupby('store_id')['units_sold'].transform(lambda x: x.shift(1).rolling(7).mean())
```

---

### 4. Random K-Fold Splitting on Temporal or Group-Structured Data
* **What happens**: Using standard `KFold(shuffle=True)` or `train_test_split()` on time-series telemetry or multi-record customer logs.
* **Why it leaks**: The model trains on tomorrow's records and predicts yesterday's records, or trains on Customer A's afternoon transaction and tests on Customer A's morning transaction.
* **The Fix**:
  - For Time-Series: Use `TimeSeriesSplit` or fixed chronological cutoffs (e.g. Train: 2023-01 to 2023-10; Test: 2023-11 to 2023-12).
  - For Grouped Entities: Use `GroupKFold` or `GroupShuffleSplit` on `customer_id` or `machine_id`.

---

### 5. Post-Event & Downstream Target Artifacts
* **What happens**: Including features that are only collected *after* the target event has occurred.
* **Examples**:
  - `account_closed_reason` in an attrition dataset (this is recorded only when attrition happens).
  - `call_center_resolution_time` in a customer complaint escalation model.
  - `chargeback_dispute_filed` in a transaction fraud model.
* **The Fix**: Conduct strict temporal grain audit: *"At the exact microsecond the model must make an inference, is this feature physically known and recorded in the database?"* If no, drop it immediately.

---

## Leakage Prevention Checklist

| Phase | Audit Question | Correct Implementation |
|---|---|---|
| **Data Ingestion** | Does this column occur temporally after the prediction moment? | Drop all downstream post-outcome columns. |
| **Data Splitting** | Is there a time dimension or entity group structure? | Use Chronological split or `GroupKFold(groups=entity_id)`. |
| **Imputation** | Are mean/median values computed on test data? | Compute summary statistics on `X_train` only; use `SimpleImputer` inside `Pipeline`. |
| **Encoding** | Does target encoding leak individual row labels? | Use Out-Of-Fold smoothed `TargetEncoder(cv=5)`. |
| **Feature Creation** | Do rolling averages or aggregations contain row $t$? | Always apply `.shift(1)` before `.rolling()`. |
| **Scikit-Learn Pipeline** | Are preprocessing and modeling unified? | Wrap all transformers in `sklearn.pipeline.Pipeline`. |
