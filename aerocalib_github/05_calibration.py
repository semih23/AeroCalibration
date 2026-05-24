#!/usr/bin/env python3
"""Applies Platt scaling and evaluates calibration metrics."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy.special import expit, logit
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression

from config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    ECE_N_BINS, RANDOM_SEED, get_feature_columns,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

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

def platt_scaling(y_val, scores_val, y_test, scores_test):
    scores_val = np.clip(scores_val, 1e-10, 1 - 1e-10)
    logits_val = logit(scores_val)
    
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(logits_val.reshape(-1, 1), y_val)
    
    A = lr.coef_[0][0]
    B = lr.intercept_[0]
    
    scores_test = np.clip(scores_test, 1e-10, 1 - 1e-10)
    logits_test = logit(scores_test)
    calibrated_test = expit(A * logits_test + B)
    
    del lr
    return calibrated_test, A, B

def compute_calibration_slope_intercept(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)
    logits = logit(y_prob)
    
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y_true)
    
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    
    del lr
    return slope, intercept

def beta_calibration(y_val, scores_val, y_test, scores_test):
    from scipy.optimize import minimize
    
    def beta_nll(params, y, s):
        a, b, c = params
        s = np.clip(s, 1e-10, 1 - 1e-10)
        
        logit_s = np.log(s / (1 - s))
        calibrated = expit(a * logit_s + b * np.log(s) + c)
        calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
        
        nll = -np.mean(y * np.log(calibrated) + (1 - y) * np.log(1 - calibrated))
        return nll
    
    result = minimize(beta_nll, [1.0, 0.0, 0.0], args=(y_val, scores_val),
                      method='L-BFGS-B', bounds=[(-10, 10), (-10, 10), (-10, 10)])
    
    a, b, c = result.x
    
    scores_test = np.clip(scores_test, 1e-10, 1 - 1e-10)
    logit_test = np.log(scores_test / (1 - scores_test))
    calibrated_test = expit(a * logit_test + b * np.log(scores_test) + c)
    
    return calibrated_test, (a, b, c)

def load_data_for_task(task_id):
    parquet_path = PROCESSED_DIR / 'aerocalib_hourly.parquet'
    
    feature_cols = get_feature_columns()
    task_config = TASKS[task_id]
    label_col = f'{task_id}_h{task_config.horizon}'
    
    needed_cols = feature_cols + [label_col, 'datetime', 'year']
    
    df = pd.read_parquet(parquet_path, columns=needed_cols)
    
    if 'year' not in df.columns:
        df['year'] = df['datetime'].dt.year
    
    val_mask = (df['year'] >= VAL_YEARS[0]) & (df['year'] <= VAL_YEARS[1])
    test_mask = (df['year'] >= TEST_YEARS[0]) & (df['year'] <= TEST_YEARS[1])
    
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    del df
    gc.collect()
    
    return val_df, test_df

def calibrate_task(task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    feature_cols = get_feature_columns()
    
    print(f"\n{'='*60}")
    print(f"Calibrating {task_id}: {task_config.name}")
    print(f"{'='*60}")
    
    model_path = MODELS_DIR / f'xgb_{task_id}_h{horizon}.json'
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None
    
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Load data
    val_df, test_df = load_data_for_task(task_id)
    
    available_features = [f for f in feature_cols if f in val_df.columns]

    val_data = val_df.dropna(subset=[label_col])
    test_data = test_df.dropna(subset=[label_col])
    
    X_val = val_data[available_features].values.astype(np.float32)
    y_val = val_data[label_col].values.astype(np.float32)
    X_test = test_data[available_features].values.astype(np.float32)
    y_test = test_data[label_col].values.astype(np.float32)
    
    del val_df, test_df, val_data, test_data
    gc.collect()
    
    # Predictions
    y_prob_val_raw = model.predict_proba(X_val)[:, 1].astype(np.float32)
    y_prob_test_raw = model.predict_proba(X_test)[:, 1].astype(np.float32)
    
    del model, X_val, X_test
    gc.collect()
    
    # Pre-calibration metrics
    ece_pre = compute_ece(y_test, y_prob_test_raw, ECE_N_BINS, 'quantile')
    brier_pre = brier_score_loss(y_test, y_prob_test_raw)
    y_prob_clipped = np.clip(y_prob_test_raw, 1e-10, 1 - 1e-10)
    nll_pre = log_loss(y_test, y_prob_clipped)
    
    print(f"  Pre-calibration (test set):")
    print(f"    ECE: {ece_pre:.6f}")
    print(f"    Brier: {brier_pre:.6f}")
    print(f"    NLL: {nll_pre:.6f}")
    
    # Platt scaling
    y_prob_test_platt, A, B = platt_scaling(y_val, y_prob_val_raw, y_test, y_prob_test_raw)
    
    ece_post = compute_ece(y_test, y_prob_test_platt, ECE_N_BINS, 'quantile')
    brier_post = brier_score_loss(y_test, y_prob_test_platt)
    y_prob_clipped_post = np.clip(y_prob_test_platt, 1e-10, 1 - 1e-10)
    nll_post = log_loss(y_test, y_prob_clipped_post)
    
    slope_post, intercept_post = compute_calibration_slope_intercept(y_test, y_prob_test_platt)
    
    print(f"  Post-calibration (Platt, test set):")
    print(f"    ECE: {ece_post:.6f}")
    print(f"    Brier: {brier_post:.6f}")
    print(f"    NLL: {nll_post:.6f}")
    print(f"    Slope: {slope_post:.3f}, Intercept: {intercept_post:.3f}")
    
    # Beta calibration
    y_prob_test_beta, beta_params = beta_calibration(y_val, y_prob_val_raw, y_test, y_prob_test_raw)
    nll_beta = log_loss(y_test, np.clip(y_prob_test_beta, 1e-10, 1 - 1e-10))
    delta_nll_beta = nll_beta - nll_post
    
    print(f"  Beta calibration delta NLL vs Platt: {delta_nll_beta:.6f}")
    
    # Cleanup
    del y_val, y_prob_val_raw, y_prob_test_platt, y_prob_test_beta
    gc.collect()
    
    return {
        'task': task_id,
        'horizon': horizon,
        'n_test': int(len(y_test)),
        'prevalence': float(y_test.mean()),
        'pre_calibration': {
            'ece_q10': float(ece_pre),
            'brier': float(brier_pre),
            'nll': float(nll_pre)
        },
        'post_calibration': {
            'ece_q10': float(ece_post),
            'brier': float(brier_post),
            'nll': float(nll_post),
            'slope': float(slope_post),
            'intercept': float(intercept_post)
        },
        'platt_params': {
            'A': float(A),
            'B': float(B)
        },
        'beta_calibration': {
            'delta_nll_vs_platt': float(delta_nll_beta)
        }
    }

def main():
    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '05_calibration.py',
            'latex_table': 'Table 4 (Calibration Summary)',
            'ece_config': {
                'n_bins': ECE_N_BINS,
                'strategy': 'quantile (equal-frequency)'
            }
        },
        'tasks': {}
    }
    
    for task_id in TASKS.keys():
        print_memory(f"before {task_id}")
        result = calibrate_task(task_id)
        if result:
            all_results['tasks'][task_id] = result
        gc.collect()
        print_memory(f"after {task_id}")
    
    output_json = RESULTS_DIR / 'calibration.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_json}")
    
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
if __name__ == "__main__":
    main()
