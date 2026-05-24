#!/usr/bin/env python3
"""Computes cost-optimal thresholds for different FN:FP cost ratios."""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import gc

warnings.filterwarnings('ignore')

import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR,
    TASKS, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    COST_RATIOS, RANDOM_SEED, get_feature_columns,
    clear_memory, print_memory
)

np.random.seed(RANDOM_SEED)

def calibrate_probabilities(y_val, prob_val, prob_test):
    prob_val = np.clip(prob_val, 1e-10, 1 - 1e-10)
    logits_val = logit(prob_val)
    
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(logits_val.reshape(-1, 1), y_val)
    
    prob_test = np.clip(prob_test, 1e-10, 1 - 1e-10)
    logits_test = logit(prob_test)
    calibrated = expit(lr.coef_[0][0] * logits_test + lr.intercept_[0])
    
    del lr
    return calibrated

def find_optimal_threshold(y_true, y_prob, cost_ratio):
    thresholds = np.linspace(0.001, 0.999, 1000)
    best_threshold = 0.5
    min_cost = float('inf')
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        cost = fn * cost_ratio + fp * 1
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = thresh
    
    return best_threshold

def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'recall': float(recall), 'precision': float(precision), 'fpr': float(fpr)
    }

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

def analyze_task(task_id):
    task_config = TASKS[task_id]
    horizon = task_config.horizon
    label_col = f'{task_id}_h{horizon}'
    feature_cols = get_feature_columns()
    
    print(f"\n{'='*60}")
    print(f"Cost-Benefit Analysis for {task_id}")
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
    y_prob_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
    y_prob_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
    
    del model, X_val, X_test
    gc.collect()
    
    # Calibrate
    y_prob_test_cal = calibrate_probabilities(y_val, y_prob_val, y_prob_test)
    y_prob_val_cal = calibrate_probabilities(y_val, y_prob_val, y_prob_val)
    
    results = {
        'task': task_id, 'horizon': horizon,
        'n_test': int(len(y_test)), 'n_positive': int(y_test.sum()),
        'prevalence': float(y_test.mean()), 'cost_ratios': {}
    }
    
    for cost_ratio in COST_RATIOS:
        print(f"\n  Cost ratio FN:FP = {cost_ratio}:1")
        
        threshold = find_optimal_threshold(y_val, y_prob_val_cal, cost_ratio)
        print(f"    Optimal threshold (val): {threshold:.3f}")
        
        metrics = evaluate_at_threshold(y_test, y_prob_test_cal, threshold)
        expected_cost = metrics['fn'] * cost_ratio + metrics['fp'] * 1
        baseline_cost = y_test.sum() * cost_ratio
        cost_reduction = (baseline_cost - expected_cost) / baseline_cost if baseline_cost > 0 else 0
        
        print(f"    Test: TP={metrics['tp']}, FP={metrics['fp']}, Recall={metrics['recall']:.1%}")
        
        results['cost_ratios'][f'{cost_ratio}:1'] = {
            'threshold': float(threshold), 'metrics': metrics,
            'expected_cost': float(expected_cost), 'baseline_cost': float(baseline_cost),
            'cost_reduction_pct': float(cost_reduction * 100)
        }
    
    del y_val, y_test, y_prob_val, y_prob_test, y_prob_val_cal, y_prob_test_cal
    gc.collect()
    
    return results

def main():
    all_results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'script': '09_cost_benefit.py',
            'latex_table': 'Table 8 (Cost-Optimal Thresholds)',
            'cost_ratios_evaluated': COST_RATIOS
        },
        'tasks': {}
    }
    
    for task_id in TASKS.keys():
        print_memory(f"before {task_id}")
        result = analyze_task(task_id)
        if result:
            all_results['tasks'][task_id] = result
        gc.collect()
        print_memory(f"after {task_id}")
    
    output_json = RESULTS_DIR / 'cost_benefit.json'
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_json}")
    
    print("\nCOST-BENEFIT ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
