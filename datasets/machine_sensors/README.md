# Dataset 05: Industrial Machine Sensors

## 1. Problem & Objective
- **Problem**: Predictive maintenance and telemetry failure monitoring.
- **ML Objectives**:
  1. **Time-Series / Anomaly Classification**: Predict machine `failure` (0 = Normal, 1 = Breakdown).
  2. **Early Degradation Detection**: Detect leading indicators of mechanical wear.

## 2. Row Granularity
Each row represents 1-hour aggregated sensor readings from an industrial machine.

## 3. Data Dictionary

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `timestamp` | String (Timestamp) | Telemetry reading hour | `2023-01-01 00:00:00` to `2024-03-25 23:00:00` |
| `machine_id` | Categorical | Machine identifier (10 distinct units) | `M_01` to `M_10` |
| `temperature` | Float | Core operating temperature (°C) | `55.2` to `115.8`, `NaN` (1.5% missing) |
| `vibration` | Float | Tri-axial vibration amplitude (mm/s RMS) | `0.85` to `9.42`, `NaN` (1.5% missing) |
| `pressure` | Float | Hydraulic system pressure (bar) | `68.5` to `122.4`, `NaN` (1.5% missing) |
| `rpm` | Integer | Rotational speed of spindle | `2650` to `3450` |
| `operating_hours` | Integer | Cumulative run hours since commissioning | `1` to `10,800` |
| `maintenance_count` | Integer | Historical maintenance service count | `0` to `6` |
| `machine_age_years` | Float | Age of equipment in years | `1.5` to `8.5` |
| `load_percentage` | Float | Current motor capacity utilization (%) | `20.0%` to `100.0%` |
| `failure` | Binary (0/1) | **Target**: 1 = Critical failure occurred during hour | `0` (99.6%), `1` (0.4%) |

## 4. Intentionally Embedded Real-World Patterns
1. **Machine-Specific Baselines**: Machine `M_01` has a naturally higher thermal operating baseline than `M_04`, requiring entity-level standardization.
2. **Pre-Failure Degradation Signature**: 12-24 hours prior to failure, `vibration` increases exponentially, `temperature` drifts upward, and `pressure` destabilizes.
3. **Temporal Drift & Wear Factor**: Sensors gradually drift as hours accumulate since last maintenance.
4. **Telemetry Dropouts**: Random intermittent missing sensor values (1-2 hours) typical of industrial IoT hardware.
5. **Temporal Leakage Trap**: Shuffling rows randomly leaks temporal continuity; splitting must be strictly chronological.
