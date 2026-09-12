"""
Data Generator for Dataset 03: Customer Transactions / Fraud Detection
Problem: Predict whether a transaction is fraudulent (Imbalanced Classification)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_fraud_dataset(seed=42, n_samples=85000):
    np.random.seed(seed)
    
    n_customers = 12000
    customer_ids = [f'CUST_{c:06d}' for c in range(1, n_customers + 1)]
    
    # Customer profiles
    cust_profiles = {}
    for cid in customer_ids:
        age = int(np.clip(np.random.normal(42, 14), 18, 85))
        account_age = int(np.clip(np.random.exponential(600) + 30, 30, 3650))
        # Wealth / spending tiers: 80% regular, 15% upper-middle, 5% high net worth
        tier = np.random.choice(['regular', 'affluent', 'hnw'], p=[0.80, 0.15, 0.05])
        avg_spend = {
            'regular': np.random.uniform(25, 120),
            'affluent': np.random.uniform(150, 450),
            'hnw': np.random.uniform(600, 2500)
        }[tier]
        prev_fraud = 1 if np.random.random() < 0.02 else 0
        cust_profiles[cid] = {
            'age': age,
            'account_age_days': account_age,
            'avg_spend': avg_spend,
            'tier': tier,
            'prev_fraud': prev_fraud
        }
        
    merchant_categories = ['Grocery', 'Dining', 'Travel', 'Electronics', 'Online Retail', 'Gas & Fuel', 'Entertainment', 'Luxury Goods', 'Cryptocurrency']
    device_types = ['Mobile iOS', 'Mobile Android', 'Desktop Chrome', 'Desktop Safari', 'Tablet', 'Unknown Bot']
    locations = ['Domestic Urban', 'Domestic Suburban', 'Domestic Rural', 'International Online', 'International In-Person']
    
    rows = []
    base_time = pd.to_datetime('2024-03-01 00:00:00')
    
    for i in range(1, n_samples + 1):
        cid = np.random.choice(customer_ids)
        c_info = cust_profiles[cid]
        
        # Determine fraud status (~1.8% rate)
        # Latent risk factors
        is_fraud = 0
        fraud_prob = 0.012
        if c_info['prev_fraud'] == 1:
            fraud_prob += 0.04
            
        is_fraud = 1 if np.random.random() < fraud_prob else 0
        
        # Transaction time
        if is_fraud:
            # Fraud skewed towards night hours (01:00 - 05:00)
            hour = int(np.random.choice([0, 1, 2, 3, 4, 5, 22, 23, 14, 15], p=[0.12, 0.18, 0.20, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03, 0.03]))
        else:
            # Normal transactions peak in afternoon/evening
            hour = int(np.clip(np.random.normal(14, 4.5), 6, 23)) % 24
            
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        day_offset = np.random.randint(0, 90)
        txn_time = (base_time + pd.Timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Transaction count in last 24h
        if is_fraud:
            txn_count_24h = int(np.random.choice([1, 4, 7, 12, 18, 25], p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]))
            failed_attempts = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]))
        else:
            txn_count_24h = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.06, 0.02]))
            failed_attempts = int(np.random.choice([0, 1, 2], p=[0.94, 0.05, 0.01]))
            
        # Merchant category
        if is_fraud:
            merchant = np.random.choice(merchant_categories, p=[0.02, 0.03, 0.18, 0.25, 0.22, 0.02, 0.03, 0.15, 0.10])
            location = np.random.choice(locations, p=[0.15, 0.10, 0.05, 0.50, 0.20])
            device = np.random.choice(device_types, p=[0.15, 0.15, 0.25, 0.10, 0.05, 0.30])
        else:
            merchant = np.random.choice(merchant_categories, p=[0.30, 0.22, 0.08, 0.10, 0.15, 0.08, 0.05, 0.01, 0.01])
            location = np.random.choice(locations, p=[0.55, 0.30, 0.10, 0.04, 0.01])
            device = np.random.choice(device_types, p=[0.42, 0.35, 0.12, 0.07, 0.03, 0.01])
            
        # Transaction Amount
        avg_s = c_info['avg_spend']
        if is_fraud:
            # Fraudulent transactions are often 4x to 25x the typical spend or test $1 auth
            if np.random.random() < 0.10:
                txn_amount = round(np.random.uniform(1.0, 3.5), 2)  # Card probing test
            else:
                multiplier = np.random.uniform(4.0, 18.0)
                txn_amount = round(avg_s * multiplier + np.random.exponential(150), 2)
        else:
            # Normal log-normal variation around user mean
            txn_amount = round(max(2.0, np.random.lognormal(mean=np.log(avg_s), sigma=0.45)), 2)
            
        # Realistic missingness
        cust_age = c_info['age'] if np.random.random() > 0.04 else np.nan
        device_final = device if np.random.random() > 0.03 else np.nan
        
        rows.append({
            'transaction_id': f'TXN_{i:07d}',
            'customer_id': cid,
            'transaction_amount': txn_amount,
            'transaction_time': txn_time,
            'customer_age': cust_age,
            'account_age_days': c_info['account_age_days'],
            'transaction_count_24h': txn_count_24h,
            'average_transaction_amount': round(c_info['avg_spend'], 2),
            'merchant_category': merchant,
            'device_type': device_final,
            'location': location,
            'previous_fraud_count': c_info['prev_fraud'],
            'failed_attempts': failed_attempts,
            'is_fraud': is_fraud
        })
        
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'fraud'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_fraud_dataset()
    csv_path = out_dir / 'transaction_fraud.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated Fraud dataset with {len(df):,} rows -> {csv_path} (Fraud rate: {df['is_fraud'].mean():.2%})")
