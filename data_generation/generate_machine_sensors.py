"""
Data Generator for Dataset 05: Industrial Machine Sensors
Problem: Predict machine failure / identify abnormal machines (Time-Series & Sensor Anomaly)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_machine_sensors_dataset(seed=42, n_machines=10, n_hours=10800):
    np.random.seed(seed)
    
    start_time = pd.to_datetime('2023-01-01 00:00:00')
    timestamps = [start_time + pd.Timedelta(hours=h) for h in range(n_hours)]
    
    machine_configs = {
        f'M_{m:02d}': {
            'age_years': np.random.choice([1.5, 3.0, 4.5, 6.0, 8.5]),
            'temp_base': np.random.uniform(62.0, 72.0),
            'vib_base': np.random.uniform(1.2, 2.0),
            'press_base': np.random.uniform(98.0, 106.0),
            'rpm_base': np.random.uniform(2800, 3200),
            'wear_rate': np.random.uniform(0.00015, 0.00035)
        } for m in range(1, n_machines + 1)
    }
    
    rows = []
    
    for m_id, cfg in machine_configs.items():
        operating_hours = 0
        maintenance_count = 0
        hours_since_maint = 0
        
        # Determine failure events (every 1800 - 3200 hours)
        next_failure_hour = np.random.randint(1600, 2600)
        
        for h_idx, ts in enumerate(timestamps):
            operating_hours += 1
            hours_since_maint += 1
            
            # Daily diurnal cycle + factory load cycle
            hour_of_day = ts.hour
            is_peak = 1 if 8 <= hour_of_day <= 18 else 0
            load_pct = np.clip(np.random.normal(85 if is_peak else 55, 10), 20, 100)
            
            # Base sensor dynamics
            load_factor = (load_pct / 100.0)
            wear_factor = 1.0 + cfg['wear_rate'] * hours_since_maint
            
            temp = cfg['temp_base'] + (load_pct - 50) * 0.18 * wear_factor + np.random.normal(0, 1.2)
            vib = cfg['vib_base'] * (load_factor ** 0.8) * wear_factor + np.random.normal(0, 0.12)
            press = cfg['press_base'] + (load_factor - 0.7) * 8.0 + np.random.normal(0, 1.5)
            rpm = cfg['rpm_base'] * (0.95 + 0.08 * load_factor) + np.random.normal(0, 25)
            
            # Check failure precursor (within 24 hours of failure)
            hours_until_fail = next_failure_hour - h_idx
            is_failure = 0
            
            if 0 < hours_until_fail <= 24:
                # Exponential degradation
                surge = np.exp((24 - hours_until_fail) / 6.0)
                vib += surge * 0.45
                temp += surge * 2.2
                press -= surge * 1.8
                rpm += np.random.normal(0, surge * 15)
                
            if hours_until_fail == 0:
                is_failure = 1
                # Trigger maintenance
                maintenance_count += 1
                hours_since_maint = 0
                next_failure_hour = h_idx + np.random.randint(1800, 3200)
                
            # Missing sensor readings (telemetry dropouts)
            temp_final = temp if np.random.random() > 0.015 else np.nan
            vib_final = vib if np.random.random() > 0.015 else np.nan
            press_final = press if np.random.random() > 0.015 else np.nan
            
            rows.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'machine_id': m_id,
                'temperature': round(temp_final, 2) if pd.notnull(temp_final) else np.nan,
                'vibration': round(vib_final, 3) if pd.notnull(vib_final) else np.nan,
                'pressure': round(press_final, 2) if pd.notnull(press_final) else np.nan,
                'rpm': int(rpm),
                'operating_hours': operating_hours,
                'maintenance_count': maintenance_count,
                'machine_age_years': cfg['age_years'],
                'load_percentage': round(load_pct, 1),
                'failure': is_failure
            })
            
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'machine_sensors'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_machine_sensors_dataset()
    csv_path = out_dir / 'industrial_sensors.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated Machine Sensors dataset with {len(df):,} rows -> {csv_path} (Failure events: {df['failure'].sum()})")
