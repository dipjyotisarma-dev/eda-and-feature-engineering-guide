"""
Data Generator for Dataset 02: Used Car Market
Problem: Predict fair used-car price (Regression)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_used_cars_dataset(seed=42, n_samples=22000):
    np.random.seed(seed)
    
    brands_data = {
        'Toyota': {
            'models': ['Corolla', 'Camry', 'RAV4', 'Highlander', 'Yaris', 'Prius'],
            'base_price': 24000,
            'depreciation_rate': 0.08,
            'luxury': False
        },
        'Honda': {
            'models': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Fit'],
            'base_price': 23000,
            'depreciation_rate': 0.085,
            'luxury': False
        },
        'Ford': {
            'models': ['F-150', 'Mustang', 'Explorer', 'Escape', 'Focus', 'Fiesta'],
            'base_price': 26000,
            'depreciation_rate': 0.11,
            'luxury': False
        },
        'BMW': {
            'models': ['3-Series', '5-Series', 'X3', 'X5', 'M3', '7-Series'],
            'base_price': 52000,
            'depreciation_rate': 0.14,
            'luxury': True
        },
        'Mercedes-Benz': {
            'models': ['C-Class', 'E-Class', 'GLC', 'GLE', 'S-Class', 'A-Class'],
            'base_price': 55000,
            'depreciation_rate': 0.145,
            'luxury': True
        },
        'Hyundai': {
            'models': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Accent'],
            'base_price': 20000,
            'depreciation_rate': 0.10,
            'luxury': False
        },
        'Porsche': {
            'models': ['911', 'Cayenne', 'Macan', 'Panamera', 'Boxster', 'Rare_Classic_Speedster'],
            'base_price': 90000,
            'depreciation_rate': 0.07,
            'luxury': True
        },
        'Nissan': {
            'models': ['Altima', 'Sentra', 'Rogue', 'Pathfinder', 'Versa', 'GT-R'],
            'base_price': 21000,
            'depreciation_rate': 0.115,
            'luxury': False
        }
    }
    
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Miami', 'Seattle', 'Dallas', 'Denver', 'Atlanta']
    fuel_types = ['Gasoline', 'Diesel', 'Hybrid', 'Electric']
    transmissions = ['Automatic', 'Manual', 'CVT']
    
    current_year = 2024
    rows = []
    
    brand_keys = list(brands_data.keys())
    brand_weights = [0.22, 0.18, 0.16, 0.12, 0.11, 0.12, 0.02, 0.07]
    
    for i in range(1, n_samples + 1):
        brand = np.random.choice(brand_keys, p=brand_weights)
        b_info = brands_data[brand]
        
        # Model selection (with rare model occurrence for Porsche)
        if brand == 'Porsche' and np.random.random() < 0.02:
            model = 'Rare_Classic_Speedster'
        else:
            model = np.random.choice(b_info['models'])
            
        year = int(np.clip(np.random.normal(2017, 4.5), 2002, 2024))
        age = current_year - year
        registration_year = year if np.random.random() > 0.015 else (year - 1 if np.random.random() < 0.5 else year + 1)
        
        # Mileage roughly correlated with age (approx 12,000 miles/year + lognormal variation)
        annual_mileage = np.random.lognormal(mean=9.3, sigma=0.45)  # ~11,000 avg
        mileage = int(max(500, age * annual_mileage + np.random.uniform(0, 3000)))
        
        # Engine CC based on luxury and vehicle category
        if 'F-150' in model or 'Mustang' in model or 'GT-R' in model or '911' in model:
            engine_cc = int(np.random.choice([3500, 4000, 5000, 3800]))
        elif b_info['luxury']:
            engine_cc = int(np.random.choice([2000, 2500, 3000, 3500, 4400]))
        else:
            engine_cc = int(np.random.choice([1200, 1500, 1800, 2000, 2400, 2500]))
            
        fuel = np.random.choice(fuel_types, p=[0.72, 0.12, 0.11, 0.05])
        transmission = np.random.choice(transmissions, p=[0.78, 0.12, 0.10])
        
        # Owners
        owner_prob = [0.55, 0.28, 0.12, 0.05] if age <= 5 else [0.20, 0.40, 0.25, 0.15]
        owner_count = int(np.random.choice([1, 2, 3, 4], p=owner_prob))
        
        # Service history & Accident history
        service_history = np.random.choice(['Full', 'Partial', 'None'], p=[0.60, 0.30, 0.10])
        accident_history = np.random.choice(['None', 'Minor', 'Major'], p=[0.75, 0.18, 0.07])
        location = np.random.choice(locations)
        
        # Price calculation with nonlinear decay and multipliers
        base_p = b_info['base_price']
        if model in ['F-150', 'Mustang', 'GT-R', 'S-Class', '7-Series', '911']:
            base_p *= 1.45
            
        # Non-linear exponential depreciation with age and mileage
        deprec = (1.0 - b_info['depreciation_rate']) ** age
        mileage_penalty = np.exp(-0.0000045 * mileage)
        
        service_mult = {'Full': 1.05, 'Partial': 0.95, 'None': 0.82}[service_history]
        accident_mult = {'None': 1.0, 'Minor': 0.88, 'Major': 0.68}[accident_history]
        owner_mult = 1.0 - (owner_count - 1) * 0.04
        fuel_mult = 1.12 if fuel == 'Hybrid' else (1.08 if fuel == 'Electric' else 1.0)
        
        price = base_p * deprec * mileage_penalty * service_mult * accident_mult * owner_mult * fuel_mult
        price = price * (1.0 + (engine_cc - 2000) / 10000.0)
        price = max(1200.0, price + np.random.normal(0, price * 0.08))
        
        # Deliberate imperfections & missingness
        # 7% missing service_history (often correlated with None/Partial)
        final_service = np.nan if np.random.random() < 0.07 else service_history
        # 4% missing accident_history
        final_accident = np.nan if np.random.random() < 0.04 else accident_history
        
        # 3 intentional anomalies across the dataset for teaching outlier detection
        if i == 142:
            mileage = -500  # Erroneous negative value
        elif i == 890:
            engine_cc = 0   # Missing/zero entry
        elif i == 1420:
            mileage = 850000 # Ultra high delivery fleet mileage
        elif i == 2315:
            price = 380000 # Rare collector Porsche 911 GT3 RS
            
        rows.append({
            'car_id': f'CAR_{i:06d}',
            'brand': brand,
            'model': model,
            'year': year,
            'registration_year': registration_year,
            'mileage': mileage,
            'engine_cc': engine_cc,
            'fuel_type': fuel,
            'transmission': transmission,
            'owner_count': owner_count,
            'service_history': final_service,
            'accident_history': final_accident,
            'location': location,
            'selling_price': round(price, 2)
        })
        
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'used_cars'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_used_cars_dataset()
    csv_path = out_dir / 'used_cars.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated Used Cars dataset with {len(df):,} rows -> {csv_path}")
