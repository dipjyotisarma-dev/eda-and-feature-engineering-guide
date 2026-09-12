"""
Data Generator for Dataset 01: Retail Sales & Inventory
Problem: Predict stockout risk / understand product demand
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_retail_dataset(seed=42, n_stores=8, n_products=15, n_days=500):
    np.random.seed(seed)
    
    start_date = pd.to_datetime('2023-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_days)]
    
    categories = ['Electronics', 'Grocery', 'Apparel', 'Home & Kitchen', 'Beauty']
    product_pool = []
    for p_id in range(1, n_products + 1):
        cat = categories[(p_id - 1) % len(categories)]
        base_price = {
            'Electronics': np.random.uniform(80, 400),
            'Grocery': np.random.uniform(3, 25),
            'Apparel': np.random.uniform(20, 90),
            'Home & Kitchen': np.random.uniform(15, 120),
            'Beauty': np.random.uniform(10, 60)
        }[cat]
        product_pool.append({
            'product_id': f'P{p_id:03d}',
            'product_category': cat,
            'base_price': base_price,
            'base_demand': np.random.uniform(15, 80)
        })
    
    stores = [f'STORE_{s:02d}' for s in range(1, n_stores + 1)]
    store_multipliers = {s: np.random.uniform(0.7, 1.4) for s in stores}
    
    rows = []
    
    for s in stores:
        s_mult = store_multipliers[s]
        for prod in product_pool:
            p_id = prod['product_id']
            cat = prod['product_category']
            base_p = prod['base_price']
            base_d = prod['base_demand']
            
            curr_inventory = int(np.random.uniform(100, 300))
            lead_time = int(np.random.choice([2, 3, 5, 7], p=[0.3, 0.4, 0.2, 0.1]))
            reorder_point = int(base_d * s_mult * (lead_time + 1.5))
            order_pending_days = 0
            
            for d in dates:
                day_of_week = d.day_name()
                is_weekend = 1 if d.weekday() >= 5 else 0
                is_holiday = 1 if (d.month == 12 and d.day >= 20) or (d.month == 11 and 23 <= d.day <= 27) or (d.month == 7 and d.day == 4) else 0
                
                # Temperature seasonal curve + noise
                day_of_year = d.dayofyear
                temperature = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + np.random.normal(0, 3)
                
                # Rainfall (zero-inflated)
                has_rain = np.random.binomial(1, 0.25)
                rainfall = np.random.exponential(8.0) if has_rain else 0.0
                
                # Promotion logic
                promotion = 1 if (np.random.random() < 0.12 or is_holiday) else 0
                discount = np.random.choice([0.10, 0.20, 0.30]) if promotion else 0.0
                
                price = round(base_p * (1.0 - discount), 2)
                
                # Demand calculation with elasticity, weekend, holiday, promo, weather
                demand_mean = base_d * s_mult
                demand_mean *= (1.0 + 0.45 * promotion)
                demand_mean *= (1.0 + 0.30 * is_weekend)
                demand_mean *= (1.0 + 0.50 * is_holiday)
                if cat == 'Grocery' and rainfall > 15:
                    demand_mean *= 1.15
                elif cat == 'Apparel' and rainfall > 15:
                    demand_mean *= 0.85
                    
                # Poisson / Negative Binomial style demand with occasional spike
                actual_demand = np.random.poisson(max(1.0, demand_mean))
                if np.random.random() < 0.01:  # 1% chance bulk buying spike
                    actual_demand = int(actual_demand * np.random.uniform(2.5, 4.0))
                
                # Sales cannot exceed current inventory
                units_sold = min(curr_inventory, actual_demand)
                stockout_occurred = 1 if actual_demand > curr_inventory else 0
                
                # Inventory update
                curr_inventory -= units_sold
                
                # Reorder logic
                if curr_inventory <= reorder_point and order_pending_days == 0:
                    order_pending_days = lead_time
                
                if order_pending_days > 0:
                    order_pending_days -= 1
                    if order_pending_days == 0:
                        replenishment = int(base_d * s_mult * 10)
                        curr_inventory += replenishment
                
                # Stockout risk label for ML: will stockout happen in next 3 days?
                stockout_risk = 1 if curr_inventory < (demand_mean * 2.0) else 0
                
                rows.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'store_id': s,
                    'product_id': p_id,
                    'product_category': cat,
                    'price': price,
                    'discount': discount if np.random.random() > 0.04 else np.nan,  # 4% MCAR missingness
                    'promotion': promotion,
                    'units_sold': units_sold,
                    'inventory': curr_inventory,
                    'supplier_lead_time': lead_time,
                    'holiday': is_holiday,
                    'weekday': day_of_week,
                    'temperature': round(temperature, 1),
                    'rainfall': round(rainfall, 2) if np.random.random() > 0.06 else np.nan,  # 6% missingness
                    'stockout_risk': stockout_risk
                })
                
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'retail'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_retail_dataset()
    csv_path = out_dir / 'retail_sales_inventory.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated Retail dataset with {len(df):,} rows -> {csv_path}")
