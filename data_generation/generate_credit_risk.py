"""
Data Generator for Dataset 06: Loan / Credit Risk
Problem: Predict probability of loan default (Classification)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_credit_risk_dataset(seed=42, n_samples=52000):
    np.random.seed(seed)
    
    home_ownerships = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase', 'medical', 'education']
    
    rows = []
    
    for i in range(1, n_samples + 1):
        # Income with right-skewed lognormal distribution
        income = int(np.clip(np.random.lognormal(mean=11.1, sigma=0.55), 18000, 450000))
        
        # Credit score (FICO style, bounded between 350 and 850)
        credit_score = int(np.clip(np.random.normal(680, 75), 380, 850))
        
        # Employment length in years
        emp_len = round(float(np.clip(np.random.exponential(6.0), 0, 35)), 1)
        
        # Home ownership & Dependents
        home = np.random.choice(home_ownerships, p=[0.42, 0.46, 0.11, 0.01])
        dependents = int(np.random.choice([0, 1, 2, 3, 4], p=[0.48, 0.22, 0.18, 0.08, 0.04]))
        purpose = np.random.choice(purposes, p=[0.45, 0.22, 0.12, 0.08, 0.06, 0.04, 0.03])
        
        # Loan term (36 or 60 months)
        loan_term = int(np.random.choice([36, 60], p=[0.70, 0.30]))
        
        # Loan amount proportional to income but capped
        max_borrow = income * 0.45
        loan_amount = int(np.clip(np.random.lognormal(mean=9.5, sigma=0.65), 2000, min(65000, max_borrow * 1.5)))
        
        # Interest rate based on credit score + term + noise
        base_rate = 6.0 + (850 - credit_score) * 0.035 + (2.0 if loan_term == 60 else 0.0)
        interest_rate = round(float(np.clip(base_rate + np.random.normal(0, 1.2), 4.5, 32.0)), 2)
        
        # Existing debt
        existing_debt = int(np.clip(np.random.exponential(income * 0.22), 0, income * 1.8))
        
        # Monthly payment calculation: standard amortized payment formula
        r = (interest_rate / 100.0) / 12.0
        n = loan_term
        monthly_payment = round(loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1), 2)
        
        # Default probability logic
        dti = (existing_debt / 12.0 + monthly_payment) / (income / 12.0)
        
        # Risk score calculation
        z = -2.8
        z += (700 - credit_score) * 0.012
        z += (dti - 0.35) * 4.5
        z += (interest_rate - 12.0) * 0.12
        if loan_term == 60:
            z += 0.30
        if home == 'RENT':
            z += 0.25
        if purpose == 'small_business':
            z += 0.40
            
        prob_default = 1.0 / (1.0 + np.exp(-z))
        default = 1 if np.random.random() < prob_default else 0
        
        # Realistic missingness
        emp_final = emp_len if np.random.random() > 0.07 else np.nan  # 7% missing (gig / self-employed)
        debt_final = existing_debt if np.random.random() > 0.03 else np.nan # 3% missing
        
        rows.append({
            'applicant_id': f'APP_{i:06d}',
            'income': income,
            'loan_amount': loan_amount,
            'loan_term_months': loan_term,
            'interest_rate': interest_rate,
            'credit_score': credit_score,
            'employment_length_years': emp_final,
            'existing_debt': debt_final,
            'monthly_payment': monthly_payment,
            'dependents': dependents,
            'home_ownership': home,
            'loan_purpose': purpose,
            'default': default
        })
        
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'credit_risk'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_credit_risk_dataset()
    csv_path = out_dir / 'loan_default.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated Credit Risk dataset with {len(df):,} rows -> {csv_path} (Default rate: {df['default'].mean():.2%})")
