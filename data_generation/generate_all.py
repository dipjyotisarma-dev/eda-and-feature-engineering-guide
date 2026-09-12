"""
Master script to generate all 6 synthetic datasets reproducibly.
"""

import sys
import time
from pathlib import Path

# Add current folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_retail import generate_retail_dataset
from generate_used_cars import generate_used_cars_dataset
from generate_fraud import generate_fraud_dataset
from generate_workforce import generate_workforce_dataset
from generate_machine_sensors import generate_machine_sensors_dataset
from generate_credit_risk import generate_credit_risk_dataset

def main():
    base_dir = Path(__file__).resolve().parent.parent / 'datasets'
    
    generators = [
        ('Retail Sales & Inventory', 'retail', 'retail_sales_inventory.csv', generate_retail_dataset),
        ('Used Car Market', 'used_cars', 'used_cars.csv', generate_used_cars_dataset),
        ('Customer Transactions & Fraud', 'fraud', 'transaction_fraud.csv', generate_fraud_dataset),
        ('Employee Workforce & Attrition', 'workforce', 'employee_attrition.csv', generate_workforce_dataset),
        ('Industrial Machine Sensors', 'machine_sensors', 'industrial_sensors.csv', generate_machine_sensors_dataset),
        ('Loan & Credit Risk', 'credit_risk', 'loan_default.csv', generate_credit_risk_dataset),
    ]
    
    print("=" * 70)
    print("GENERATING SYNTHETIC DATASETS FOR EDA & FEATURE ENGINEERING GUIDE")
    print("=" * 70)
    
    start_total = time.time()
    
    for name, folder, filename, gen_fn in generators:
        out_folder = base_dir / folder
        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / filename
        
        t0 = time.time()
        print(f"\n[+] Generating {name}...")
        df = gen_fn()
        df.to_csv(out_path, index=False)
        dur = time.time() - t0
        print(f"    --> Saved to {out_path}")
        print(f"    --> Shape: {df.shape[0]:,} rows x {df.shape[1]} columns ({dur:.2f}s)")
        
    print("\n" + "=" * 70)
    print(f"ALL DATASETS GENERATED SUCCESSFULLY IN {time.time() - start_total:.2f}s!")
    print("=" * 70)

if __name__ == '__main__':
    main()
