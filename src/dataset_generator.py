import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

def generate_datasets():
    print("[1/3] Generating synthetic reference and production datasets...")
    np.random.seed(42)
    
    # 10 informative features simulating a credit risk / financial fraud model
    feature_names = [
        "credit_score_scaled",
        "debt_to_income_ratio",
        "monthly_income_log",
        "revolving_utilization",
        "num_credit_inquiries_6m",
        "delinquency_history_count",
        "loan_amount_requested",
        "employment_duration_years",
        "savings_balance_scaled",
        "transaction_velocity_score"
    ]
    
    # Baseline Reference Data (e.g. historical training distribution)
    X_ref, y_ref = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.8, 0.2], # 20% default rate
        random_state=42
    )
    
    ref_df = pd.DataFrame(X_ref, columns=feature_names)
    ref_df["target"] = y_ref
    
    # Current Batch 1: Stable live traffic (No Drift, standard noise)
    X_cur_stable, y_cur_stable = make_classification(
        n_samples=1500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=101
    )
    cur_stable_df = pd.DataFrame(X_cur_stable, columns=feature_names)
    cur_stable_df["target"] = y_cur_stable
    
    # Current Batch 2: Severe Covariate & Concept Drift
    # Simulate an economic shift (e.g. inflation, policy change):
    # - Debt-to-income ratio shifts higher (+1.8 sigma)
    # - Revolving utilization spikes (+1.5 sigma)
    # - Credit inquiries jump (+1.2 sigma)
    # - Savings balance drops (-1.4 sigma)
    cur_drift_df = cur_stable_df.copy()
    cur_drift_df["debt_to_income_ratio"] += np.random.normal(1.8, 0.5, size=len(cur_drift_df))
    cur_drift_df["revolving_utilization"] += np.random.normal(1.5, 0.4, size=len(cur_drift_df))
    cur_drift_df["num_credit_inquiries_6m"] += np.random.normal(1.2, 0.3, size=len(cur_drift_df))
    cur_drift_df["savings_balance_scaled"] -= np.random.normal(1.4, 0.4, size=len(cur_drift_df))
    
    # Target shift (defaults jump from 20% to ~38%)
    prob_shift = 1 / (1 + np.exp(-(cur_drift_df["debt_to_income_ratio"] + cur_drift_df["revolving_utilization"])))
    cur_drift_df["target"] = (prob_shift > 0.6).astype(int)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    ref_df.to_csv(os.path.join(data_dir, "reference_train.csv"), index=False)
    cur_stable_df.to_csv(os.path.join(data_dir, "batch_stable.csv"), index=False)
    cur_drift_df.to_csv(os.path.join(data_dir, "batch_drifted.csv"), index=False)
    
    print(f" Datasets saved successfully in: {data_dir}")
    print(f" - Reference data: {ref_df.shape}")
    print(f" - Stable batch: {cur_stable_df.shape}")
    print(f" - Drifted batch: {cur_drift_df.shape}")

if __name__ == "__main__":
    generate_datasets()
