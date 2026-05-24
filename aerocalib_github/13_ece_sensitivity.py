#!/usr/bin/env python3
"""
ECE Bin Sensitivity Analysis
Çalıştır: python 13_ece_sensitivity.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.special import expit, logit
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# Paths - config.py'den
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

TASKS = {
    'T1': {'horizon': 6},
    'T2': {'horizon': 6},
    'T3': {'horizon': 6},
    'T4': {'horizon': 1}
}

def get_feature_columns():
    features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'dayofweek_sin', 'dayofweek_cos', 'dayofyear']
    lag_vars = ['wind_speed', 'wind_dir', 'wind_u', 'wind_v', 
                'wind_dir_sin', 'wind_dir_cos',
                'visibility', 'ceiling', 'temperature', 'dewpoint', 'pressure']
    for var in lag_vars:
        for lag in [1, 2, 3, 6, 12, 24]:
            features.append(f'{var}_lag{lag}')
    for var in ['has_fog', 'has_precip', 'has_gust']:
        for lag in [1, 2, 3, 6, 12, 24]:
            features.append(f'{var}_lag{lag}')
    for var in ['wind_u', 'wind_v', 'wind_speed']:
        for window in [6, 24]:
            features.append(f'{var}_roll{window}_mean')
            features.append(f'{var}_roll{window}_std')
    features.extend(['lat', 'lon', 'rwy_hdg'])
    return features

def compute_ece(y_true, y_prob, n_bins=10, strategy='quantile'):
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins = np.unique(bins)
    else:
        bins = np.linspace(0, 1, n_bins + 1)
    
    bin_indices = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() * np.abs(avg_accuracy - avg_confidence)
    return ece / len(y_true)

def platt_scaling(y_val, scores_val, scores_test):
    scores_val = np.clip(scores_val, 1e-10, 1 - 1e-10)
    logits_val = logit(scores_val)
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(logits_val.reshape(-1, 1), y_val)
    scores_test = np.clip(scores_test, 1e-10, 1 - 1e-10)
    logits_test = logit(scores_test)
    return expit(lr.coef_[0][0] * logits_test + lr.intercept_[0])

def main():
    print("Loading data...")
    df = pd.read_parquet(PROCESSED_DIR / 'aerocalib_hourly.parquet')
    df['year'] = df['datetime'].dt.year
    
    val_df = df[(df['year'] >= 2021) & (df['year'] <= 2022)]
    test_df = df[(df['year'] >= 2023) & (df['year'] <= 2024)]
    
    feature_cols = get_feature_columns()
    results = {}
    
    for task_id, task_config in TASKS.items():
        horizon = task_config['horizon']
        label_col = f'{task_id}_h{horizon}'
        model_path = MODELS_DIR / f'xgb_{task_id}_h{horizon}.json'
        
        if not model_path.exists():
            print(f"  {task_id}: Model not found")
            continue
        
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        
        available_features = [f for f in feature_cols if f in val_df.columns]
        
        val_data = val_df.dropna(subset=[label_col])
        test_data = test_df.dropna(subset=[label_col])
        
        X_val = val_data[available_features].values
        y_val = val_data[label_col].values
        X_test = test_data[available_features].values
        y_test = test_data[label_col].values
        
        y_prob_val = model.predict_proba(X_val)[:, 1]
        y_prob_test_raw = model.predict_proba(X_test)[:, 1]
        y_prob_test_cal = platt_scaling(y_val, y_prob_val, y_prob_test_raw)
        
        print(f"\n{task_id}:")
        task_results = {}
        for n_bins in [10, 15, 20]:
            ece_pre = compute_ece(y_test, y_prob_test_raw, n_bins, 'quantile')
            ece_post = compute_ece(y_test, y_prob_test_cal, n_bins, 'quantile')
            task_results[f'Q{n_bins}'] = {'pre': ece_pre, 'post': ece_post}
            print(f"  Q{n_bins}: pre={ece_pre:.6f}, post={ece_post:.6f}")
        
        results[task_id] = task_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Max difference between Q10 and Q15/Q20 (post-cal)")
    for task_id, task_res in results.items():
        q10 = task_res['Q10']['post']
        q15 = task_res['Q15']['post']
        q20 = task_res['Q20']['post']
        max_diff = max(abs(q10-q15), abs(q10-q20))
        print(f"  {task_id}: Q10={q10:.6f}, max_diff={max_diff:.6f}")
    
    # Save
    with open(RESULTS_DIR / 'ece_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'ece_sensitivity.json'}")

if __name__ == "__main__":
    main()
