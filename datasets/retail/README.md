# Dataset 01: Retail Sales & Inventory

## 1. Problem & Objective
- **Problem**: Multi-store retail inventory optimization and demand forecasting.
- **ML Objectives**:
  1. **Classification**: Predict binary `stockout_risk` (whether inventory will deplete within supplier lead time).
  2. **Regression**: Predict continuous `units_sold` (demand forecasting).

## 2. Row Granularity
Each row represents the daily activity of a single `product_id` at a specific `store_id` on a specific `date`.

## 3. Data Dictionary

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `date` | String (YYYY-MM-DD) | Date of observation | `2023-01-01` to `2024-05-14` |
| `store_id` | Categorical | Unique store identifier (8 stores) | `STORE_01`, `STORE_08` |
| `product_id` | Categorical | Unique product identifier (15 products) | `P001`, `P015` |
| `product_category` | Categorical | High-level category | `Electronics`, `Grocery`, `Apparel`, `Home & Kitchen`, `Beauty` |
| `price` | Float | Effective selling price after discount ($) | `3.50` to `380.00` |
| `discount` | Float | Promotional discount percentage (0.0 to 0.30) | `0.0`, `0.10`, `0.20`, `0.30`, `NaN` (4% missing) |
| `promotion` | Binary (0/1) | Whether an active marketing campaign is running | `0`, `1` |
| `units_sold` | Integer | Daily sales volume (bounded by inventory) | `0` to `280` |
| `inventory` | Integer | Stock on hand at day start | `0` to `850` |
| `supplier_lead_time` | Integer | Days required for replenishment to arrive | `2`, `3`, `5`, `7` |
| `holiday` | Binary (0/1) | Major shopping holiday flag | `0`, `1` |
| `weekday` | Categorical | Day of the week | `Monday` to `Sunday` |
| `temperature` | Float | Average outdoor temperature (°C) | `-8.5` to `38.2` |
| `rainfall` | Float | Daily precipitation in mm | `0.0` to `48.5`, `NaN` (6% missing) |
| `stockout_risk` | Binary (0/1) | **Target for Classification**: 1 if stock is critically low | `0`, `1` |

## 4. Intentionally Embedded Real-World Patterns
1. **Weekly & Seasonal Demand**: Weekend uplift (+30%), holiday surges (+50%), and sinusoidal temperature cycles.
2. **Promotion Lift**: Promotions increase demand by ~45% while decreasing price.
3. **Category Weather Sensitivity**: Rainy days increase grocery demand (+15%) and decrease apparel store traffic (-15%).
4. **Inventory Truncation**: When stock runs out, `units_sold` caps at `inventory` (censored demand problem).
5. **Missing Values**: Missingness in `rainfall` and `discount` to test imputation strategies.
6. **Rare Spikes**: Occasional 1% bulk purchases creating right-tail demand outliers.
