# Dataset 04: Employee Workforce & Attrition

## 1. Problem & Objective
- **Problem**: Proactive workforce retention and turnover prediction.
- **ML Objective**:
  - **Binary Classification**: Predict `attrition` (0 = Stayed, 1 = Left Company).

## 2. Row Granularity
Each row represents a full-time employee record at the annual review cycle.

## 3. Data Dictionary

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `employee_id` | String | Unique staff identifier | `EMP_00001` |
| `age` | Integer | Employee age in years | `21` to `64` |
| `department` | Categorical | Business unit (5 departments) | `Engineering`, `Sales`, `HR`, `Product & Design`, `Operations` |
| `job_role` | Categorical | Specific functional title | `Software Engineer`, `Account Executive`, `HR Specialist`, etc. |
| `salary` | Integer | Annual base compensation ($) | `$48,000` to `$240,000` |
| `years_at_company` | Integer | Total company tenure | `0` to `35` |
| `years_in_role` | Integer | Tenure in current position | `0` to `22` (always `<= years_at_company`) |
| `overtime` | Categorical | Exempt overtime status | `Yes`, `No` |
| `satisfaction_score` | Integer (Ordinal) | Employee survey sentiment | `1` (Extremely Dissatisfied) to `5` (Extremely Satisfied) |
| `commute_distance` | Float | One-way distance to office (miles) | `1.0` to `55.0`, `NaN` (6% missing) |
| `promotion_count` | Integer | Number of career promotions received | `0` to `8` |
| `training_hours` | Float | Professional development hours completed | `0` to `80`, `NaN` (4% missing) |
| `performance_score` | Integer (Ordinal) | Most recent performance rating | `1` to `5` |
| `attrition` | Binary (0/1) | **Target**: 1 = Voluntarily Left, 0 = Retained | `0` (83.8%), `1` (16.2%) |

## 4. Intentionally Embedded Real-World Patterns
1. **Interaction Effects**: `overtime == 'Yes'` compounded by `satisfaction_score <= 2` produces severe attrition risk.
2. **Career Stagnation Signals**: High `years_in_role` (>= 4) with 0 promotions is a strong attrition predictor.
3. **Multicollinearity / Redundancy**: `years_at_company` and `years_in_role` have high collinearity.
4. **Compensation Inequity**: Salary relative to department/role benchmark is more predictive than raw salary alone.
5. **Right-Skewed Compensation**: Highly skewed salary distribution across executive vs operational roles.
