"""
Data Generator for Dataset 04: Employee / Workforce
Problem: Predict employee attrition (Classification)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_workforce_dataset(seed=42, n_samples=16000):
    np.random.seed(seed)
    
    departments = {
        'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'QA Analyst', 'DevOps Specialist'],
        'Sales': ['Sales Representative', 'Account Executive', 'Sales Director'],
        'Human Resources': ['HR Specialist', 'Recruiter', 'HR Director'],
        'Product & Design': ['Product Manager', 'UX Designer', 'Product Analyst'],
        'Operations': ['Operations Analyst', 'Supply Chain Lead', 'Customer Support Specialist']
    }
    
    base_salaries = {
        'Software Engineer': 85000, 'Senior Engineer': 135000, 'Tech Lead': 165000,
        'QA Analyst': 72000, 'DevOps Specialist': 110000,
        'Sales Representative': 58000, 'Account Executive': 95000, 'Sales Director': 175000,
        'HR Specialist': 62000, 'Recruiter': 58000, 'HR Director': 145000,
        'Product Manager': 120000, 'UX Designer': 88000, 'Product Analyst': 78000,
        'Operations Analyst': 65000, 'Supply Chain Lead': 92000, 'Customer Support Specialist': 48000
    }
    
    dept_keys = list(departments.keys())
    dept_weights = [0.38, 0.25, 0.08, 0.14, 0.15]
    
    rows = []
    
    for i in range(1, n_samples + 1):
        dept = np.random.choice(dept_keys, p=dept_weights)
        role = np.random.choice(departments[dept])
        
        age = int(np.clip(np.random.normal(36, 9), 21, 64))
        
        # Max tenure bounded by age - 20
        max_possible_tenure = max(1, age - 20)
        years_at_company = int(np.clip(np.random.exponential(4.5), 0.5, max_possible_tenure))
        
        # Years in role <= years_at_company
        years_in_role = int(min(years_at_company, np.clip(np.random.exponential(2.8), 0, years_at_company)))
        
        # Promotions
        if years_at_company <= 2:
            promotion_count = 0
        else:
            promotion_count = int(np.random.poisson(years_at_company / 3.5))
            
        # Overtime
        ot_prob = 0.35 if dept in ['Sales', 'Engineering'] else 0.18
        overtime = 'Yes' if np.random.random() < ot_prob else 'No'
        
        # Satisfaction score (1 to 5)
        satisfaction_score = int(np.random.choice([1, 2, 3, 4, 5], p=[0.12, 0.18, 0.32, 0.26, 0.12]))
        
        # Performance score (1 to 5)
        perf_score = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.12, 0.55, 0.22, 0.06]))
        
        # Commute distance in miles
        commute = round(float(np.clip(np.random.exponential(11.0) + 1, 1, 55)), 1)
        
        # Training hours in last year
        training_hours = int(np.clip(np.random.normal(28, 12), 0, 80))
        
        # Salary calculation
        role_base = base_salaries[role]
        tenure_bonus = 1.0 + (years_at_company * 0.032)
        perf_bonus = 1.0 + (perf_score - 3) * 0.05
        market_variation = np.random.normal(1.0, 0.08)
        salary = int(round(role_base * tenure_bonus * perf_bonus * market_variation, -2))
        
        # Attrition logic (realistic probability formula)
        # Log-odds of leaving
        z = -2.2
        if overtime == 'Yes':
            z += 0.85
        if satisfaction_score <= 2:
            z += (3 - satisfaction_score) * 0.75
        if satisfaction_score >= 4:
            z -= 0.50
        if commute > 25:
            z += 0.40
        if years_in_role >= 4 and promotion_count == 0:
            z += 0.65  # Stagnant career
        if salary < (role_base * 0.92):
            z += 0.50  # Underpaid
        if age < 28:
            z += 0.35  # Higher job-hopping among early career
            
        prob_attrition = 1.0 / (1.0 + np.exp(-z))
        attrition = 1 if np.random.random() < prob_attrition else 0
        
        # Missingness
        commute_final = commute if np.random.random() > 0.06 else np.nan
        training_final = training_hours if np.random.random() > 0.04 else np.nan
        
        rows.append({
            'employee_id': f'EMP_{i:05d}',
            'age': age,
            'department': dept,
            'job_role': role,
            'salary': salary,
            'years_at_company': years_at_company,
            'years_in_role': years_in_role,
            'overtime': overtime,
            'satisfaction_score': satisfaction_score,
            'commute_distance': commute_final,
            'promotion_count': promotion_count,
            'training_hours': training_final,
            'performance_score': perf_score,
            'attrition': attrition
        })
        
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    out_dir = Path(__file__).resolve().parent.parent / 'datasets' / 'workforce'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_workforce_dataset()
    csv_path = out_dir / 'employee_attrition.csv'
    df.to_csv(csv_path, index=False)
    print(f"Generated Workforce dataset with {len(df):,} rows -> {csv_path} (Attrition rate: {df['attrition'].mean():.2%})")
