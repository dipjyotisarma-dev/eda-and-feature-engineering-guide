# Dataset 02: Used Car Market

## 1. Problem & Objective
- **Problem**: Fair market valuation of pre-owned vehicles.
- **ML Objective**:
  - **Regression**: Predict continuous `selling_price` ($).

## 2. Row Granularity
Each row represents an individual used vehicle listed for sale.

## 3. Data Dictionary

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `car_id` | String | Unique vehicle listing identifier | `CAR_000001` |
| `brand` | Categorical | Vehicle manufacturer (8 brands) | `Toyota`, `Honda`, `Ford`, `BMW`, `Porsche`, etc. |
| `model` | Categorical | Vehicle model | `Corolla`, `3-Series`, `911`, `F-150`, etc. |
| `year` | Integer | Model manufacture year | `2002` to `2024` |
| `registration_year` | Integer | Year registered with DMV | `2001` to `2024` (contains 1-2 edge discrepancies) |
| `mileage` | Integer | Total odometer distance (miles) | `500` to `350,000` (contains deliberate anomalies) |
| `engine_cc` | Integer | Engine displacement in cubic centimeters | `1200` to `5000` |
| `fuel_type` | Categorical | Primary fuel | `Gasoline`, `Diesel`, `Hybrid`, `Electric` |
| `transmission` | Categorical | Transmission type | `Automatic`, `Manual`, `CVT` |
| `owner_count` | Integer | Number of previous registered owners | `1`, `2`, `3`, `4` |
| `service_history` | Categorical | Maintenance record completeness | `Full`, `Partial`, `None`, `NaN` (7% missing) |
| `accident_history` | Categorical | Reported accident severity | `None`, `Minor`, `Major`, `NaN` (4% missing) |
| `location` | Categorical | Metropolitan sales region (10 cities) | `New York`, `Los Angeles`, `Chicago`, etc. |
| `selling_price` | Float | **Target**: Final listing transaction price ($) | `$1,200` to `$165,000` (with rare collector outliers) |

## 4. Intentionally Embedded Real-World Patterns
1. **Exponential Depreciation**: Vehicle value drops non-linearly with age (`(1 - r)^age`) and mileage (`exp(-k * mileage)`).
2. **Right-Skewed Target**: `selling_price` is heavily right-skewed, requiring evaluation of log1p transformations for linear models.
3. **Category-Specific Multipliers**: Luxury brands (Porsche, BMW, Mercedes) depreciate faster in absolute terms but maintain high base values.
4. **Rare Categories**: Exotic models (e.g. `Rare_Classic_Speedster`) with < 10 instances.
5. **Data Quality Defects**:
   - Negative mileage data-entry bug (`mileage = -500`)
   - Zero displacement engine (`engine_cc = 0`)
   - Ultra-high commercial delivery mileage outlier (850,000 miles)
   - Exotic collector price outlier ($380,000)
